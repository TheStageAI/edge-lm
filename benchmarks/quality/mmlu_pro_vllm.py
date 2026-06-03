"""Production MMLU-Pro runner for Gemma 4 release benchmarks.

This is the public, standalone version of the vLLM MMLU-Pro protocol used for
the release tables. It intentionally evaluates standard Hugging Face BF16
checkpoints, not MLX runtime artifacts, so compressed checkpoints and public
baselines can be compared through the same backend.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Any


CHOICES = list("ABCDEFGHIJKLMNOP")

DEFAULT_SUBJECTS = [
    "biology",
    "business",
    "chemistry",
    "computer science",
    "economics",
    "engineering",
    "health",
    "history",
    "law",
    "math",
    "philosophy",
    "physics",
    "psychology",
    "other",
]

INITIAL_PROMPT = (
    'The following are multiple choice questions (with answers) about {$}. '
    'Think step by step and then finish your answer with "the answer is (X)" '
    "where X is the correct letter choice.\n"
)


def normalize_subject(subject: str) -> str:
    return subject.strip().replace("_", " ").lower()


def subject_slug(subject: str) -> str:
    return normalize_subject(subject).replace(" ", "_")


def run_mmlu_pro_subject(
    *,
    model_path: str,
    output_dir: str | Path,
    subject: str,
    backend: str = "vllm",
    ntrain: int = 5,
    max_model_len: int = 4096,
    max_new_tokens: int = 2048,
    gpu_memory_utilization: float = 0.85,
    tensor_parallel_size: int | None = None,
    trust_remote_code: bool = True,
    enable_prefix_caching: bool = True,
    enforce_eager: bool = True,
    prompt_format: str = "raw",
    enable_thinking: bool = False,
    system_prompt: str | None = None,
    stop_sequences: list[str] | None = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = -1,
    seed: int | None = None,
    transformers_batch_size: int = 4,
) -> dict[str, Any]:
    """Run the TIGER-Lab MMLU-Pro local-vLLM protocol for one subject.

    This intentionally mirrors the official repository's local evaluator:
    vLLM, CoT prompts, and answer extraction from "the answer is (X)" style
    outputs. The per-subject shard writes raw predictions; final official-style
    accuracy should be computed by aggregating all subject files in sorted
    subject order.
    """

    import torch
    import transformers
    from datasets import load_dataset
    from transformers import AutoProcessor

    runtime_model_path = _materialize_hf_model(model_path)
    backend = backend.lower().strip()
    if backend not in {"vllm", "transformers"}:
        raise ValueError(f"Unsupported backend={backend!r}")
    prompt_format = prompt_format.lower().strip()
    if prompt_format not in {"raw", "gemma4_chat"}:
        raise ValueError(f"Unsupported prompt_format={prompt_format!r}")
    if stop_sequences is None:
        stop_sequences = ["Question:"] if prompt_format == "raw" else []
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    slug = subject_slug(subject)
    subject_output = output_path / f"{slug}.json"
    subject_summary = output_path / f"{slug}.summary.json"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.info("loading MMLU-Pro dataset")
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    full_test = _preprocess(dataset["test"])
    full_val = _preprocess(dataset["validation"])

    all_subjects = sorted({normalize_subject(row["category"]) for row in full_test})
    target_subject = _resolve_subject(subject, all_subjects)
    test_df = _select_by_category(full_test, target_subject)
    val_df = _select_by_category(full_val, target_subject)
    if not test_df:
        raise ValueError(f"No MMLU-Pro test rows found for subject={subject!r}; subjects={all_subjects}")
    if not val_df:
        raise ValueError(f"No MMLU-Pro validation rows found for subject={subject!r}; subjects={all_subjects}")

    if tensor_parallel_size is None:
        tensor_parallel_size = max(1, torch.cuda.device_count())

    logging.info(
        "loading vLLM model=%s subject=%s rows=%d ntrain=%d max_model_len=%d max_new_tokens=%d tp=%d prompt_format=%s enable_thinking=%s stop_sequences=%s temperature=%.4f top_p=%.4f top_k=%d seed=%s",
        runtime_model_path,
        target_subject,
        len(test_df),
        ntrain,
        max_model_len,
        max_new_tokens,
        tensor_parallel_size,
        prompt_format,
        enable_thinking,
        stop_sequences,
        temperature,
        top_p,
        top_k,
        seed,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        runtime_model_path,
        trust_remote_code=trust_remote_code,
    )
    processor = None
    if prompt_format == "gemma4_chat":
        processor = AutoProcessor.from_pretrained(
            runtime_model_path,
            trust_remote_code=trust_remote_code,
        )

    prompts: list[str] = []
    prompt_lengths: list[int] = []
    effective_k: list[int] = []
    for row in test_df:
        prompt, length, k_used = _build_prompt_with_length_guard(
            tokenizer=tokenizer,
            processor=processor,
            val_df=val_df,
            row=row,
            ntrain=ntrain,
            max_model_len=max_model_len,
            max_new_tokens=max_new_tokens,
            prompt_format=prompt_format,
            enable_thinking=enable_thinking,
            system_prompt=system_prompt,
        )
        prompts.append(prompt)
        prompt_lengths.append(length)
        effective_k.append(k_used)

    start = time.time()
    if backend == "vllm":
        from vllm import LLM, SamplingParams

        llm_kwargs: dict[str, Any] = {
            "model": runtime_model_path,
            "gpu_memory_utilization": gpu_memory_utilization,
            "tensor_parallel_size": tensor_parallel_size,
            "max_model_len": max_model_len,
            "trust_remote_code": trust_remote_code,
            "enforce_eager": enforce_eager,
        }
        if enable_prefix_caching:
            llm_kwargs["enable_prefix_caching"] = True
        llm = LLM(**llm_kwargs)
        sampling_kwargs = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_new_tokens,
            "stop": stop_sequences,
        }
        if seed is not None:
            sampling_kwargs["seed"] = seed
        try:
            sampling_params = SamplingParams(**sampling_kwargs)
        except TypeError:
            sampling_kwargs.pop("seed", None)
            sampling_params = SamplingParams(**sampling_kwargs)
        logging.info("running vLLM generate subject=%s prompts=%d", target_subject, len(prompts))
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        generated_records = [
            {
                "text": output.outputs[0].text,
                "finish_reason": getattr(output.outputs[0], "finish_reason", None),
                "stop_reason": getattr(output.outputs[0], "stop_reason", None),
                "generated_token_count": len(getattr(output.outputs[0], "token_ids", []) or []),
            }
            for output in outputs
        ]
    else:
        logging.info(
            "running Transformers generate subject=%s prompts=%d batch_size=%d",
            target_subject,
            len(prompts),
            transformers_batch_size,
        )
        generated_records = [
            {
                "text": text,
                "finish_reason": None,
                "stop_reason": None,
                "generated_token_count": None,
            }
            for text in _generate_with_transformers(
            model_path=runtime_model_path,
            prompts=prompts,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            stop_sequences=stop_sequences,
            trust_remote_code=trust_remote_code,
            batch_size=transformers_batch_size,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            )
        ]
    elapsed = time.time() - start

    results: list[dict[str, Any]] = []
    for row, generated_record, prompt_len, k_used in zip(
        test_df,
        generated_records,
        prompt_lengths,
        effective_k,
        strict=True,
    ):
        generated_text = str(generated_record["text"])
        pred = extract_answer(generated_text)
        item = dict(row)
        item["pred"] = pred
        item["model_outputs"] = generated_text
        item["finish_reason"] = generated_record["finish_reason"]
        item["stop_reason"] = generated_record["stop_reason"]
        item["generated_token_count"] = generated_record["generated_token_count"]
        item["prompt_length"] = prompt_len
        item["effective_ntrain"] = k_used
        results.append(_jsonable(item))

    subject_output.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    stats = score_results(results, official_unknown_fallback=False)
    summary = {
        "model_path": model_path,
        "runtime_model_path": runtime_model_path,
        "backend": backend,
        "subject": target_subject,
        "subject_slug": subject_slug(target_subject),
        "ntrain": ntrain,
        "max_model_len": max_model_len,
        "max_new_tokens": max_new_tokens,
        "gpu_memory_utilization": gpu_memory_utilization,
        "tensor_parallel_size": tensor_parallel_size,
        "trust_remote_code": trust_remote_code,
        "enable_prefix_caching": enable_prefix_caching,
        "enforce_eager": enforce_eager,
        "prompt_format": prompt_format,
        "enable_thinking": enable_thinking,
        "system_prompt": system_prompt,
        "stop_sequences": stop_sequences,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "seed": seed,
        "transformers_batch_size": transformers_batch_size,
        "num_rows": len(results),
        "elapsed_seconds": elapsed,
        "rows_per_second": len(results) / elapsed if elapsed > 0 else None,
        "prompt_length_max": max(prompt_lengths) if prompt_lengths else None,
        "prompt_length_mean": sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else None,
        "effective_ntrain_min": min(effective_k) if effective_k else None,
        "finish_reason_counts": _count_values(row.get("finish_reason") for row in results),
        "stop_reason_counts": _count_values(row.get("stop_reason") for row in results),
        "generated_token_count_max": _max_optional(row.get("generated_token_count") for row in results),
        "generated_token_count_mean": _mean_optional(row.get("generated_token_count") for row in results),
        "metrics_without_unknown_random": stats,
        "output_file": str(subject_output),
    }
    subject_summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    logging.info("subject=%s summary=%s", target_subject, json.dumps(summary, sort_keys=True))
    return summary


def _generate_with_transformers(
    *,
    model_path: str,
    prompts: list[str],
    tokenizer: Any,
    max_new_tokens: int,
    stop_sequences: list[str],
    trust_remote_code: bool,
    batch_size: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM

    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=trust_remote_code,
    ).to("cuda")
    model.eval()
    out: list[str] = []
    device = next(model.parameters()).device
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        input_width = int(inputs["input_ids"].shape[1])
        with torch.inference_mode():
            do_sample = temperature > 0
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if do_sample:
                generation_kwargs.update(
                    {
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k,
                    }
                )
            output_ids = model.generate(
                **inputs,
                **generation_kwargs,
            )
        new_ids = output_ids[:, input_width:]
        texts = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
        out.extend(_apply_stop(text, stop_sequences) for text in texts)
    return out


def _apply_stop(text: str, stop_sequences: list[str]) -> str:
    if not stop_sequences:
        return text
    cut = len(text)
    for stop in stop_sequences:
        if not stop:
            continue
        idx = text.find(stop)
        if idx >= 0:
            cut = min(cut, idx)
    return text[:cut]


def _materialize_hf_model(model_path: str) -> str:
    """Return a local model path when `model_path` is a Hugging Face repo id.

    vLLM and Transformers can both load remote repo ids directly, but the Gemma4
    processor path currently performs extra Hub metadata calls while constructing
    the tokenizer. Materializing the repo into the shared Modal cache first makes
    the rest of the eval use a local filesystem path, so those metadata checks do
    not become per-shard failure points.
    """

    path = Path(model_path)
    if path.exists() or model_path.startswith("/"):
        return model_path
    if "/" not in model_path:
        return model_path

    from huggingface_hub import snapshot_download

    hf_home = Path(os.environ.get("HF_HOME", "/tmp/huggingface"))
    local_dir = hf_home / "tq-model-snapshots" / model_path.replace("/", "__")
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )
    last_error: BaseException | None = None
    for attempt in range(1, 4):
        try:
            logging.info(
                "materializing HF model repo=%s local_dir=%s attempt=%d",
                model_path,
                local_dir,
                attempt,
            )
            resolved = snapshot_download(
                repo_id=model_path,
                local_dir=str(local_dir),
                cache_dir=str(hf_home / "hub"),
                token=token,
                max_workers=1,
            )
            logging.info("materialized HF model repo=%s resolved=%s", model_path, resolved)
            return str(local_dir)
        except BaseException as exc:
            last_error = exc
            logging.warning(
                "HF snapshot_download failed repo=%s attempt=%d error=%s",
                model_path,
                attempt,
                exc,
            )
            time.sleep(10 * attempt)
    raise RuntimeError(f"Failed to materialize HF model {model_path}") from last_error


def aggregate_subject_outputs(
    *,
    input_dir: str | Path,
    output_path: str | Path,
    subjects: list[str] | None = None,
    official_unknown_fallback: bool = True,
) -> dict[str, Any]:
    input_path = Path(input_dir)
    if subjects is None:
        subjects = DEFAULT_SUBJECTS
    rows_by_subject: dict[str, list[dict[str, Any]]] = {}
    for subject in sorted(normalize_subject(s) for s in subjects):
        path = input_path / f"{subject_slug(subject)}.json"
        if not path.exists():
            continue
        rows_by_subject[subject] = json.loads(path.read_text(encoding="utf-8"))
    if not rows_by_subject:
        raise FileNotFoundError(f"No subject outputs found in {input_path}")

    random.seed(12345)
    by_subject: dict[str, dict[str, Any]] = {}
    total_corr = 0.0
    total_wrong = 0.0
    total_unknown = 0
    for subject in sorted(rows_by_subject):
        stats = score_results(
            rows_by_subject[subject],
            official_unknown_fallback=official_unknown_fallback,
        )
        by_subject[subject] = stats
        total_corr += float(stats["corr"])
        total_wrong += float(stats["wrong"])
        total_unknown += int(stats["unknown"])
    total_acc = total_corr / (total_corr + total_wrong + 1e-6)
    summary = {
        "accuracy": total_acc,
        "corr": total_corr,
        "wrong": total_wrong,
        "unknown": total_unknown,
        "num_subjects": len(by_subject),
        "official_unknown_fallback": official_unknown_fallback,
        "by_subject": by_subject,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def score_results(
    rows: list[dict[str, Any]],
    *,
    official_unknown_fallback: bool,
) -> dict[str, Any]:
    corr = 0.0
    wrong = 0.0
    unknown = 0
    for row in rows:
        pred = row.get("pred")
        if not pred:
            unknown += 1
            if official_unknown_fallback:
                x = random.randint(0, len(row["options"]) - 1)
                if x == int(row["answer_index"]):
                    corr += 1
                else:
                    wrong += 1
            else:
                wrong += 1
        elif pred == row.get("answer"):
            corr += 1
        else:
            wrong += 1
    acc = corr / (corr + wrong + 1e-6)
    return {
        "accu": acc,
        "corr": corr,
        "wrong": wrong,
        "unknown": unknown,
        "num_rows": len(rows),
    }


def extract_answer(text: str) -> str | None:
    pattern = r"answer is\s*:?\s*\(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return extract_again(text)


def extract_again(text: str) -> str | None:
    match = re.search(r".*[aA]nswer\s*:?\s*\(?([A-J])\)?", text)
    if match:
        return match.group(1)
    return extract_final(text)


def extract_final(text: str) -> str | None:
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    return None


def _preprocess(split: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in split:
        item = dict(row)
        item["options"] = [opt for opt in item["options"] if opt != "N/A"]
        rows.append(item)
    return rows


def _resolve_subject(subject: str, all_subjects: list[str]) -> str:
    wanted = normalize_subject(subject)
    if wanted in all_subjects:
        return wanted
    wanted_slug = subject_slug(wanted)
    matches = [s for s in all_subjects if wanted_slug in subject_slug(s)]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"Could not resolve subject={subject!r}; matches={matches}; all={all_subjects}")


def _select_by_category(rows: list[dict[str, Any]], subject: str) -> list[dict[str, Any]]:
    normalized = normalize_subject(subject)
    return [row for row in rows if normalize_subject(row["category"]) == normalized]


def _count_values(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = "null" if value is None else str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _mean_optional(values: Any) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def _max_optional(values: Any) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return max(numeric)


def _format_cot_example(row: dict[str, Any], *, including_answer: bool) -> str:
    prompt = "Question:\n"
    prompt += row["question"] + "\n"
    prompt += "Options:\n"
    for i, opt in enumerate(row["options"]):
        prompt += f"{CHOICES[i]}. {opt}\n"
    if including_answer:
        cot_content = row["cot_content"].replace(
            "A: Let's think step by step.",
            "Answer: Let's think step by step.",
        )
        prompt += cot_content + "\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    return prompt


def _generate_cot_prompt(val_df: list[dict[str, Any]], row: dict[str, Any], k: int) -> str:
    subject = row["category"]
    prompt = INITIAL_PROMPT.replace("{$}", subject) + "\n"
    for example in val_df[:k]:
        prompt += _format_cot_example(example, including_answer=True)
    prompt += _format_cot_example(row, including_answer=False)
    return prompt


def _build_prompt_with_length_guard(
    *,
    tokenizer: Any,
    processor: Any,
    val_df: list[dict[str, Any]],
    row: dict[str, Any],
    ntrain: int,
    max_model_len: int,
    max_new_tokens: int,
    prompt_format: str,
    enable_thinking: bool,
    system_prompt: str | None,
) -> tuple[str, int, int]:
    k = ntrain
    last_prompt = ""
    last_length = 0
    while k >= 0:
        prompt = _generate_cot_prompt(val_df, row, k)
        prompt = _wrap_prompt(
            prompt,
            tokenizer=tokenizer,
            processor=processor,
            prompt_format=prompt_format,
            enable_thinking=enable_thinking,
            system_prompt=system_prompt,
        )
        length = len(tokenizer(prompt, return_tensors="pt")["input_ids"][0])
        last_prompt = prompt
        last_length = length
        if length < max_model_len - max_new_tokens:
            return prompt, length, k
        k -= 1
    return last_prompt, last_length, 0


def _wrap_prompt(
    prompt: str,
    *,
    tokenizer: Any,
    processor: Any,
    prompt_format: str,
    enable_thinking: bool,
    system_prompt: str | None,
) -> str:
    if prompt_format == "raw":
        return prompt
    if prompt_format != "gemma4_chat":
        raise ValueError(f"Unsupported prompt_format={prompt_format!r}")

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    template_owner = processor if processor is not None else tokenizer
    return template_owner.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return value


def _load_protocol(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        path = Path(__file__).resolve().parent / "protocols" / "gemma4_mmlu_pro_vllm.json"
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("mmlu_pro_official", data)


def _parse_subjects(raw: str | None) -> list[str]:
    if not raw:
        return DEFAULT_SUBJECTS
    return [item.strip() for item in raw.split(",") if item.strip()]


def _build_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description="Run the production vLLM MMLU-Pro benchmark.")
    parser.add_argument(
        "--protocol",
        default=str(Path(__file__).resolve().parent / "protocols" / "gemma4_mmlu_pro_vllm.json"),
        help="JSON protocol file with mmlu_pro_official settings.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_subject = subparsers.add_parser("run-subject", help="Run one MMLU-Pro subject shard.")
    run_subject.add_argument("--model", required=True, help="HF repo id or local HF checkpoint directory.")
    run_subject.add_argument("--subject", required=True, help="Subject name, e.g. biology.")
    run_subject.add_argument("--output-dir", required=True, help="Directory for subject JSON outputs.")
    run_subject.add_argument("--backend", choices=["vllm", "transformers"], default=None)
    run_subject.add_argument("--tensor-parallel-size", type=int, default=None)

    aggregate = subparsers.add_parser("aggregate", help="Aggregate subject JSON outputs.")
    aggregate.add_argument("--input-dir", required=True, help="Directory containing <subject>.json outputs.")
    aggregate.add_argument(
        "--output",
        default=None,
        help="Aggregate JSON output path. Defaults to <input-dir>/../summary.official_random.json.",
    )
    aggregate.add_argument("--subjects", default=None, help="Comma-separated subject list.")
    aggregate.add_argument(
        "--no-official-unknown-fallback",
        action="store_true",
        help="Count missing answers as wrong instead of applying the official random fallback.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    protocol = _load_protocol(args.protocol)

    if args.command == "run-subject":
        summary = run_mmlu_pro_subject(
            model_path=args.model,
            output_dir=args.output_dir,
            subject=args.subject,
            backend=args.backend or str(protocol.get("backend", "vllm")),
            ntrain=int(protocol.get("ntrain", 0)),
            max_model_len=int(protocol.get("max_model_len", 12288)),
            max_new_tokens=int(protocol.get("max_new_tokens", 8192)),
            gpu_memory_utilization=float(protocol.get("gpu_memory_utilization", 0.85)),
            tensor_parallel_size=args.tensor_parallel_size,
            enable_prefix_caching=bool(protocol.get("enable_prefix_caching", True)),
            enforce_eager=bool(protocol.get("enforce_eager", True)),
            prompt_format=str(protocol.get("prompt_format", "gemma4_chat")),
            enable_thinking=bool(protocol.get("enable_thinking", True)),
            system_prompt=protocol.get("system_prompt"),
            stop_sequences=protocol.get("stop_sequences", []),
            temperature=float(protocol.get("temperature", 1.0)),
            top_p=float(protocol.get("top_p", 0.95)),
            top_k=int(protocol.get("top_k", 64)),
            seed=protocol.get("seed", 42),
            transformers_batch_size=int(protocol.get("transformers_batch_size", 4)),
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    if args.command == "aggregate":
        input_dir = Path(args.input_dir)
        output = Path(args.output) if args.output else input_dir.parent / "summary.official_random.json"
        summary = aggregate_subject_outputs(
            input_dir=input_dir,
            output_path=output,
            subjects=_parse_subjects(args.subjects),
            official_unknown_fallback=not args.no_official_unknown_fallback,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
