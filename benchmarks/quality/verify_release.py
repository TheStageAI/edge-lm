"""End-to-end release benchmark verifier.

This script downloads public HF artifacts, materializes them into standard
Hugging Face BF16 checkpoints, then runs the production vLLM benchmark scripts.

Supported source formats:
- TheStage MLX release checkpoints: dequantized on CPU with PyTorch from
  `model_{m,l}.safetensors` and `ple_{m,l}.safetensors`.
- Public GGUF checkpoints: dequantized with the `gguf` reader and saved back as
  standard safetensors using the Gemma 4 Hugging Face key layout.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_REGISTRY = THIS_DIR / "release_models.json"
DEFAULT_WORK_DIR = Path("runs/release_verify")
VALID_BENCHMARKS = {"mmlu_pro", "ifeval"}
METADATA_FILES = (
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
)


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _model_keys(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _subjects(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


def _max_shard_bytes(raw: str) -> int:
    text = raw.strip().upper()
    if text.endswith("GB"):
        return int(float(text[:-2]) * 1_000_000_000)
    if text.endswith("GIB"):
        return int(float(text[:-3]) * 1024**3)
    if text.endswith("MB"):
        return int(float(text[:-2]) * 1_000_000)
    return int(text)


def _checkpoint_dir(work_dir: Path, key: str) -> Path:
    return work_dir / "checkpoints" / key


def _run_dir(work_dir: Path, key: str, benchmark: str) -> Path:
    return work_dir / "evals" / key / benchmark


def _ensure_empty_dir(path: Path, *, force: bool) -> None:
    if path.exists():
        if not force:
            marker = path / "config.json"
            if marker.exists():
                return
            raise FileExistsError(f"{path} already exists; pass --force to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _has_safetensors_checkpoint(path: Path) -> bool:
    return (path / "model.safetensors").exists() or (path / "model.safetensors.index.json").exists()


def _checkpoint_ready(path: Path, spec: dict[str, Any]) -> bool:
    if not (path / "config.json").exists():
        return False
    if spec["kind"] in {"thestage_mlx", "gguf"}:
        return _has_safetensors_checkpoint(path)
    return True


def _copy_metadata_files(source_dir: Path, output_dir: Path) -> None:
    for name in METADATA_FILES:
        src = source_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)


def _copy_missing_metadata_files(source_dir: Path, output_dir: Path) -> None:
    for name in METADATA_FILES:
        dst = output_dir / name
        if dst.exists():
            continue
        src = source_dir / name
        if src.exists():
            shutil.copy2(src, dst)


def _snapshot(repo: str, *, allow_patterns: list[str] | None = None) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=repo, allow_patterns=allow_patterns))


def prepare_checkpoint(
    key: str,
    spec: dict[str, Any],
    *,
    work_dir: Path,
    force: bool,
    max_shard_size: str,
) -> Path:
    out = _checkpoint_dir(work_dir, key)
    if out.exists() and not force:
        if _checkpoint_ready(out, spec):
            print(f"{key}: using existing materialized checkpoint at {out}")
            return out
        raise FileExistsError(f"{out} exists but is incomplete; pass --force to replace it.")

    kind = spec["kind"]
    if kind == "thestage_mlx":
        materialize_thestage_mlx(spec, out, force=force, max_shard_size=max_shard_size)
    elif kind == "gguf":
        materialize_gguf(spec, out, force=force, max_shard_size=max_shard_size)
    elif kind == "hf":
        materialize_hf(spec, out, force=force)
    else:
        raise ValueError(f"Unsupported model kind for {key}: {kind!r}")
    return out


def materialize_hf(spec: dict[str, Any], out: Path, *, force: bool) -> None:
    _ensure_empty_dir(out, force=force)
    local = _snapshot(spec["repo"])
    _copy_tree(local, out)


def materialize_gguf(spec: dict[str, Any], out: Path, *, force: bool, max_shard_size: str) -> None:
    _ensure_empty_dir(out, force=force)
    from huggingface_hub import hf_hub_download

    repo = spec["repo"]
    gguf_file = spec["gguf_file"]
    base_model = spec.get("base_model")
    if not base_model:
        raise ValueError("GGUF materialization requires base_model for Gemma tokenizer/config metadata.")

    print(f"Downloading GGUF {repo}:{gguf_file}")
    downloaded = Path(hf_hub_download(repo, gguf_file))
    base_local = _snapshot(base_model, allow_patterns=list(METADATA_FILES))
    _copy_metadata_files(base_local, out)
    use_gemma4_text_only_config(out / "config.json")

    (out / "source_gguf.json").write_text(
        json.dumps(
            {
                "repo": repo,
                "gguf_file": gguf_file,
                "base_model": base_model,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Materializing GGUF {downloaded} -> {out}")
    writer = ShardedSafeTensorWriter(out, max_shard_bytes=_max_shard_bytes(max_shard_size))
    materialize_gemma4_gguf(downloaded, writer)
    writer.finish()


def _lm_eval_model_args(checkpoint: Path) -> list[str]:
    return ["--model", str(checkpoint)]


GGUF_TOP_LEVEL_KEYS = {
    "output_norm.weight": "language_model.model.norm.weight",
    "per_layer_model_proj.weight": "language_model.model.per_layer_model_projection.weight",
    "per_layer_proj_norm.weight": "language_model.model.per_layer_projection_norm.weight",
    "per_layer_token_embd.weight": "language_model.model.embed_tokens_per_layer.weight",
    "token_embd.weight": "language_model.model.embed_tokens.weight",
}

GGUF_LAYER_KEYS = {
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "attn_norm.weight": "input_layernorm.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_norm.weight": "pre_feedforward_layernorm.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "inp_gate.weight": "per_layer_input_gate.weight",
    "layer_output_scale.weight": "layer_scalar",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "post_ffw_norm.weight": "post_feedforward_layernorm.weight",
    "post_norm.weight": "post_per_layer_input_norm.weight",
    "proj.weight": "per_layer_projection.weight",
}


def _map_gemma4_gguf_key(name: str) -> str | None:
    if name == "rope_freqs.weight":
        return None
    mapped = GGUF_TOP_LEVEL_KEYS.get(name)
    if mapped is not None:
        return mapped

    parts = name.split(".")
    if len(parts) >= 4 and parts[0] == "blk":
        layer = parts[1]
        suffix = ".".join(parts[2:])
        mapped_suffix = GGUF_LAYER_KEYS.get(suffix)
        if mapped_suffix is not None:
            return f"language_model.model.layers.{layer}.{mapped_suffix}"

    raise ValueError(f"Unsupported Gemma 4 GGUF tensor key: {name}")


def _gguf_tensor_to_torch(tensor: Any) -> Any:
    import numpy as np
    import torch

    if int(tensor.tensor_type) == 0:
        array = np.asarray(tensor.data)
    else:
        import gguf.quants as quants

        array = np.asarray(quants.dequantize(tensor.data, tensor.tensor_type))

    value = torch.from_numpy(array)
    if value.dtype != torch.bfloat16:
        value = value.to(torch.bfloat16)
    return value


def materialize_gemma4_gguf(path: Path, writer: ShardedSafeTensorWriter) -> None:
    import gguf

    reader = gguf.GGUFReader(path)
    architecture = reader.fields.get("general.architecture")
    if architecture is not None:
        arch_value = architecture.contents()
        if hasattr(arch_value, "tolist"):
            arch_value = arch_value.tolist()
        if isinstance(arch_value, bytes):
            arch_value = arch_value.decode("utf-8", errors="replace")
        if arch_value != "gemma4":
            raise ValueError(f"Expected Gemma 4 GGUF architecture, got {arch_value!r}")

    converted = 0
    skipped = 0
    seen: set[str] = set()
    for index, tensor in enumerate(reader.tensors, start=1):
        target = _map_gemma4_gguf_key(tensor.name)
        if target is None:
            skipped += 1
            print(f"skipping GGUF tensor {index}/{len(reader.tensors)} {tensor.name}")
            continue
        if target in seen:
            raise ValueError(f"Duplicate mapped GGUF tensor key: {target}")
        seen.add(target)
        value = _gguf_tensor_to_torch(tensor)
        writer.add(target, value)
        converted += 1
        print(f"materialized GGUF tensor {index}/{len(reader.tensors)} {tensor.name} -> {target} {tuple(value.shape)}")

    print(f"materialized {converted} Gemma 4 GGUF tensors; skipped {skipped}")


def use_gemma4_text_only_config(config_path: Path) -> None:
    config = _load_json(config_path)
    if config.get("model_type") == "gemma4":
        config["architectures"] = ["Gemma4ForCausalLM"]
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def materialize_thestage_mlx(
    spec: dict[str, Any],
    out: Path,
    *,
    force: bool,
    max_shard_size: str,
) -> None:
    _ensure_empty_dir(out, force=force)
    size = spec.get("size", "m")
    repo = spec["repo"]
    patterns = [
        *METADATA_FILES,
        f"model_{size}.safetensors",
        f"ple_{size}.safetensors",
        "audio_tower.safetensors",
        "vision_tower.safetensors",
    ]
    local = _snapshot(repo, allow_patterns=patterns)
    _copy_metadata_files(local, out)
    base_model = spec.get("base_model")
    if base_model:
        base_local = _snapshot(base_model, allow_patterns=list(METADATA_FILES))
        _copy_missing_metadata_files(base_local, out)

    writer = ShardedSafeTensorWriter(out, max_shard_bytes=_max_shard_bytes(max_shard_size))
    alias_cache: dict[str, Any] = {}
    dequantize_mlx_safetensors(local / f"model_{size}.safetensors", writer, alias_cache=alias_cache)
    for extra_name in ("audio_tower.safetensors", "vision_tower.safetensors"):
        extra_path = local / extra_name
        if extra_path.exists():
            dequantize_mlx_safetensors(extra_path, writer, alias_cache=alias_cache)
    materialize_ple(local / f"ple_{size}.safetensors", writer)
    add_gemma4_shared_kv_aliases(out / "config.json", writer, alias_cache)
    writer.finish()


def _copy_tree(src: Path, dst: Path) -> None:
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


class ShardedSafeTensorWriter:
    def __init__(self, output_dir: Path, *, max_shard_bytes: int):
        self.output_dir = output_dir
        self.max_shard_bytes = max_shard_bytes
        self.current: dict[str, Any] = {}
        self.current_size = 0
        self.shards: list[Path] = []
        self.weight_map: dict[str, str] = {}
        self.total_size = 0

    def add(self, name: str, tensor: Any) -> None:
        size = int(tensor.numel() * tensor.element_size())
        if self.current and self.current_size + size > self.max_shard_bytes:
            self.flush()
        self.current[name] = tensor.contiguous()
        self.current_size += size
        self.total_size += size

    def flush(self) -> None:
        if not self.current:
            return
        from safetensors.torch import save_file

        shard_id = len(self.shards) + 1
        path = self.output_dir / f"model-{shard_id:05d}.safetensors"
        save_file(self.current, path, metadata={"format": "pt"})
        for key in self.current:
            self.weight_map[key] = path.name
        self.shards.append(path)
        self.current = {}
        self.current_size = 0

    def finish(self) -> None:
        self.flush()
        if len(self.shards) == 1:
            only = self.shards[0]
            final = self.output_dir / "model.safetensors"
            only.rename(final)
            for key in self.weight_map:
                self.weight_map[key] = final.name
            return
        index = {
            "metadata": {"total_size": self.total_size},
            "weight_map": self.weight_map,
        }
        (self.output_dir / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _map_mlx_key(key: str) -> str:
    if key.startswith("model.language_model."):
        return "language_model.model." + key[len("model.language_model.") :]
    if key.startswith("model.vision_tower.") or key.startswith("model.audio_tower."):
        return key[len("model.") :]
    return key


def _infer_quantization(weight: Any, scales: Any, default_bits: int | None = None, default_group_size: int | None = None) -> tuple[int, int]:
    packed_cols = int(weight.shape[-1])
    scale_cols = int(scales.shape[-1])
    bit_candidates = [default_bits] if default_bits else [2, 3, 4, 5, 6, 8]
    group_candidates = [default_group_size] if default_group_size else [16, 32, 64, 128]
    for bits in bit_candidates:
        if bits is None:
            continue
        full_cols = packed_cols * 32 // bits
        if full_cols * bits != packed_cols * 32:
            continue
        for group_size in group_candidates:
            if group_size is not None and full_cols // group_size == scale_cols:
                return int(bits), int(group_size)
    raise ValueError(
        f"Cannot infer quantization for weight={tuple(weight.shape)} scales={tuple(scales.shape)}"
    )


def _default_quantization_for_key(key: str) -> dict[str, int]:
    if any(token in key for token in ("vision_tower", "audio_tower", "embed_vision", "embed_audio")):
        return {"bits": 4}
    return {}


def _unpack_packed_u32(weight: Any, *, bits: int) -> Any:
    import torch

    packed = weight.to(torch.int64)
    rows, packed_cols = packed.shape
    full_cols = packed_cols * 32 // bits
    bit_offsets = torch.arange(full_cols, dtype=torch.int64) * bits
    word_idx = torch.div(bit_offsets, 32, rounding_mode="floor")
    bit_idx = bit_offsets % 32
    values = (packed[:, word_idx] >> bit_idx) & ((1 << bits) - 1)

    spill = bit_idx + bits - 32
    spill_mask = spill > 0
    if bool(spill_mask.any()):
        cols = torch.nonzero(spill_mask, as_tuple=False).flatten()
        next_words = torch.clamp(word_idx[cols] + 1, max=packed_cols - 1)
        high_mask = (torch.ones_like(spill[cols]) << spill[cols]) - 1
        high = packed[:, next_words] & high_mask
        values[:, cols] |= high << (bits - spill[cols])
    return values


def _dequantize_affine(weight: Any, scales: Any, biases: Any, *, bits: int, group_size: int) -> Any:
    import torch

    q = _unpack_packed_u32(weight, bits=bits)
    rows, full_cols = q.shape
    q = q.reshape(rows, full_cols // group_size, group_size).to(torch.float32)
    out = q * scales.to(torch.float32).unsqueeze(-1) + biases.to(torch.float32).unsqueeze(-1)
    return out.reshape(rows, full_cols).to(torch.bfloat16)


def _capture_shared_kv_alias_source(cache: dict[str, Any] | None, name: str, tensor: Any) -> None:
    if cache is None:
        return
    if not name.startswith("language_model.model.layers."):
        return
    if name.endswith((".self_attn.k_norm.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight")):
        cache[name] = tensor


def _convert_mlx_tensor_layout(name: str, tensor: Any) -> Any:
    if name.startswith("audio_tower.") and name.endswith(".lconv1d.depthwise_conv1d.weight") and tensor.ndim == 3:
        return tensor.transpose(1, 2)
    if name.startswith("audio_tower.") and ".conv." in name and name.endswith(".weight") and tensor.ndim == 4:
        return tensor.permute(0, 3, 1, 2)
    return tensor


def dequantize_mlx_safetensors(
    path: Path,
    writer: ShardedSafeTensorWriter,
    *,
    alias_cache: dict[str, Any] | None = None,
) -> None:
    import torch
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        quantization = json.loads(metadata.get("quantization", "{}"))
        keys = list(handle.keys())
        for key in keys:
            if key.endswith(".scales") or key.endswith(".biases"):
                continue
            mapped = _map_mlx_key(key)
            if key.endswith(".weight"):
                scales_key = key[:-7] + ".scales"
                biases_key = key[:-7] + ".biases"
                if scales_key in keys and biases_key in keys:
                    module_name = key[:-7]
                    qcfg = quantization.get(module_name, quantization)
                    if not qcfg:
                        qcfg = _default_quantization_for_key(key)
                    weight = handle.get_tensor(key)
                    scales = handle.get_tensor(scales_key)
                    biases = handle.get_tensor(biases_key)
                    bits, group_size = _infer_quantization(
                        weight,
                        scales,
                        default_bits=qcfg.get("bits") if isinstance(qcfg, dict) else None,
                        default_group_size=qcfg.get("group_size") if isinstance(qcfg, dict) else None,
                    )
                    tensor = _dequantize_affine(weight, scales, biases, bits=bits, group_size=group_size)
                    _capture_shared_kv_alias_source(alias_cache, mapped, tensor)
                    writer.add(mapped, tensor)
                    continue
            tensor = handle.get_tensor(key)
            if tensor.dtype == torch.float32:
                tensor = tensor.to(torch.bfloat16)
            tensor = _convert_mlx_tensor_layout(mapped, tensor)
            _capture_shared_kv_alias_source(alias_cache, mapped, tensor)
            writer.add(mapped, tensor)


def add_gemma4_shared_kv_aliases(
    config_path: Path,
    writer: ShardedSafeTensorWriter,
    alias_cache: dict[str, Any],
) -> None:
    if not alias_cache or not config_path.exists():
        return
    config = _load_json(config_path)
    text_config = config.get("text_config") or {}
    num_layers = int(text_config.get("num_hidden_layers") or 0)
    num_shared = int(text_config.get("num_kv_shared_layers") or 0)
    layer_types = list(text_config.get("layer_types") or [])
    if num_layers <= 0 or num_shared <= 0 or len(layer_types) != num_layers:
        return

    shared_start = max(0, num_layers - num_shared)
    last_non_shared_by_type: dict[str, int] = {}
    for layer_idx in range(num_layers):
        layer_type = str(layer_types[layer_idx])
        prefix = f"language_model.model.layers.{layer_idx}.self_attn"
        has_own_kv = f"{prefix}.k_proj.weight" in alias_cache
        if layer_idx < shared_start and has_own_kv:
            last_non_shared_by_type[layer_type] = layer_idx
            continue
        if has_own_kv:
            last_non_shared_by_type[layer_type] = layer_idx
            continue

        source_idx = last_non_shared_by_type.get(layer_type)
        if source_idx is None:
            continue
        source_prefix = f"language_model.model.layers.{source_idx}.self_attn"
        for suffix in ("k_norm.weight", "k_proj.weight", "v_proj.weight"):
            source_key = f"{source_prefix}.{suffix}"
            target_key = f"{prefix}.{suffix}"
            value = alias_cache.get(source_key)
            if value is not None and target_key not in alias_cache:
                tensor = value.detach().clone()
                writer.add(target_key, tensor)
                alias_cache[target_key] = tensor


def _unpack_packed_indices(packed: Any, shape: tuple[int, ...], bits: int, *, chunk_size: int = 8_000_000) -> Any:
    import torch

    total = math.prod(shape)
    flat = torch.empty(total, dtype=torch.uint8)
    packed_i64 = packed.to(torch.int64).flatten()
    mask = (1 << bits) - 1
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        count = stop - start
        start_bit = start * bits
        byte_start = start_bit // 8
        bit_offset = start_bit % 8
        bytes_needed = (bit_offset + count * bits + 7) // 8 + 1
        buf = packed_i64[byte_start : byte_start + bytes_needed]
        # The final code may end exactly at the end of the packed byte stream.
        # Add one zero byte so the two-byte read below is safe on the last
        # chunk without changing decoded values.
        buf = torch.cat([buf, torch.zeros(1, dtype=buf.dtype)])
        offsets = torch.arange(count, dtype=torch.int64) * bits + bit_offset
        byte_idx = torch.div(offsets, 8, rounding_mode="floor")
        bit_idx = offsets % 8
        combined = buf[byte_idx] | (buf[byte_idx + 1] << 8)
        flat[start:stop] = ((combined >> bit_idx) & mask).to(torch.uint8)
    return flat.reshape(shape)


def materialize_ple(path: Path, writer: ShardedSafeTensorWriter, *, vocab_chunk: int = 2048) -> None:
    import torch
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        codebooks = handle.get_tensor("codebooks")
        codes_packed = handle.get_tensor("codes_packed")

    stages = int(metadata["stages"])
    codes_shape = tuple(json.loads(metadata["codes_shape"]))
    bits = int(metadata["index_storage_bits"])
    codes = _unpack_packed_indices(codes_packed, codes_shape, bits)
    if stages == 1:
        codes = codes[:, :, :, 0]
        codebooks = codebooks[:, :, 0, :, :]

    num_layers = int(metadata["num_layers"])
    num_groups = int(metadata["num_groups"])
    group_size = int(metadata["group_size"])
    ple_dim = int(metadata["ple_dim"])
    vocab_size = int(metadata["vocab_size"])
    output = torch.empty((vocab_size, num_layers * ple_dim), dtype=torch.bfloat16)

    l_idx = torch.arange(num_layers).reshape(num_layers, 1, 1)
    g_idx = torch.arange(num_groups).reshape(1, num_groups, 1)
    for start in range(0, vocab_size, vocab_chunk):
        stop = min(start + vocab_chunk, vocab_size)
        chunk_codes = codes[:, :, start:stop].to(torch.long)
        gathered = codebooks[l_idx, g_idx, chunk_codes]
        output[start:stop] = gathered.permute(2, 0, 1, 3).reshape(stop - start, num_layers * num_groups * group_size)
        print(f"materialized PLE rows {start}:{stop}")

    target = _map_mlx_key(metadata.get("target", "model.language_model.embed_tokens_per_layer"))
    writer.add(f"{target}.weight", output)


def run_benchmarks(
    key: str,
    *,
    checkpoint: Path,
    work_dir: Path,
    benchmarks: set[str],
    subjects: list[str] | None,
) -> None:
    if "mmlu_pro" in benchmarks:
        out = _run_dir(work_dir, key, "mmlu_pro")
        subjects_dir = out / "subjects"
        subjects_dir.mkdir(parents=True, exist_ok=True)
        protocol = _load_json(THIS_DIR / "protocols" / "gemma4_mmlu_pro_vllm.json")
        subject_list = subjects or protocol["mmlu_pro_official"]["subjects"]
        for subject in subject_list:
            _run(
                [
                    sys.executable,
                    str(THIS_DIR / "mmlu_pro_vllm.py"),
                    "run-subject",
                    "--model",
                    str(checkpoint),
                    "--subject",
                    subject,
                    "--output-dir",
                    str(subjects_dir),
                ]
            )
        _run(
            [
                sys.executable,
                str(THIS_DIR / "mmlu_pro_vllm.py"),
                "aggregate",
                "--input-dir",
                str(subjects_dir),
                "--output",
                str(out / "summary.official_random.json"),
            ]
        )

    if "ifeval" in benchmarks:
        out = _run_dir(work_dir, key, "ifeval")
        out.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(THIS_DIR / "lm_eval_vllm.py"),
            "--output-dir",
            str(out),
        ]
        command.extend(_lm_eval_model_args(checkpoint))
        _run(command)


def _run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare public checkpoints and run release vLLM evals.")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR))
    parser.add_argument("--max-shard-size", default="4GB")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Download and materialize model checkpoints.")
    prepare.add_argument("--models", required=True, help="Comma-separated model keys from release_models.json.")
    prepare.add_argument("--force", action="store_true")

    run = subparsers.add_parser("run", help="Prepare models if needed and run vLLM benchmarks.")
    run.add_argument("--models", required=True, help="Comma-separated model keys from release_models.json.")
    run.add_argument("--benchmarks", default="mmlu_pro,ifeval", help="Comma-separated: mmlu_pro, ifeval.")
    run.add_argument("--subjects", default=None, help="Optional comma-separated MMLU-Pro subject list.")
    run.add_argument("--force-prepare", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    registry = _load_json(args.registry)
    work_dir = Path(args.work_dir)
    keys = _model_keys(args.models)

    for key in keys:
        if key not in registry:
            raise KeyError(f"Unknown model key {key!r}. Available: {', '.join(sorted(registry))}")

    if args.command == "prepare":
        for key in keys:
            prepare_checkpoint(
                key,
                registry[key],
                work_dir=work_dir,
                force=args.force,
                max_shard_size=args.max_shard_size,
            )
        return

    if args.command == "run":
        benchmarks = set(_model_keys(args.benchmarks))
        unknown = benchmarks - VALID_BENCHMARKS
        if unknown:
            raise ValueError(f"Unknown benchmark(s): {', '.join(sorted(unknown))}")
        for key in keys:
            checkpoint = prepare_checkpoint(
                key,
                registry[key],
                work_dir=work_dir,
                force=args.force_prepare,
                max_shard_size=args.max_shard_size,
            )
            run_benchmarks(
                key,
                checkpoint=checkpoint,
                work_dir=work_dir,
                benchmarks=benchmarks,
                subjects=_subjects(args.subjects),
            )
        return

    raise ValueError(args.command)


if __name__ == "__main__":
    main()
