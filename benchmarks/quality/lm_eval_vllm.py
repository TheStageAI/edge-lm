"""Run production lm-eval suites through a standard HF checkpoint + vLLM.

The release IFEval path uses this backend family rather than the local MLX
diagnostic runner. The model argument must point at a Hugging Face repo id or a
local standard Hugging Face BF16 checkpoint directory.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _load_protocol(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _model_args(
    model: str,
    *,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    add_bos_token: bool,
    enable_thinking: bool | None,
) -> str:
    parts = {
        "pretrained": model,
        "dtype": "bfloat16",
        "trust_remote_code": "True",
        "tensor_parallel_size": str(tensor_parallel_size),
        "gpu_memory_utilization": str(gpu_memory_utilization),
        "add_bos_token": str(bool(add_bos_token)),
    }
    if enable_thinking is not None:
        parts["enable_thinking"] = str(bool(enable_thinking))
    return ",".join(f"{key}={value}" for key, value in parts.items())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a vLLM-backed lm-eval protocol.")
    parser.add_argument("--model", required=True, help="HF repo id or local HF checkpoint directory.")
    parser.add_argument(
        "--protocol",
        default=str(Path(__file__).resolve().parent / "protocols" / "gemma4_ifeval_vllm.json"),
        help="JSON protocol file with lm_eval settings.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-samples", action="store_true")
    args = parser.parse_args()

    from lm_eval import evaluator

    protocol = _load_protocol(args.protocol)
    cfg = protocol["lm_eval"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    results = evaluator.simple_evaluate(
        model="vllm",
        model_args=_model_args(
            args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            add_bos_token=bool(cfg.get("add_bos_token", True)),
            enable_thinking=cfg.get("enable_thinking"),
        ),
        tasks=cfg["tasks"],
        batch_size=cfg.get("batch_size", "auto"),
        max_batch_size=cfg.get("max_batch_size"),
        log_samples=args.log_samples,
        apply_chat_template=cfg.get("apply_chat_template", True),
        fewshot_as_multiturn=cfg.get("fewshot_as_multiturn", True),
        gen_kwargs=cfg.get("gen_kwargs"),
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
    )

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2, sort_keys=True, default=str) + "\n")
    print(f"Results saved to {metrics_path}")


if __name__ == "__main__":
    main()
