# Benchmarks

Quality tests run every checkpoint through one vLLM-backed serving path.
Performance tests run the released artifacts in native MLX on Apple Silicon.

| Track | Question | Execution path | Canonical results |
| --- | --- | --- | --- |
| [Quality](quality/) | How much model behavior survives compression? | Backend-equalized Hugging Face checkpoints served through vLLM | [`quality/results/gemma4_release_quality.csv`](quality/results/gemma4_release_quality.csv) |
| [Performance](#native-mlx-performance) | How fast and memory-efficient is the deployed model? | Native `edge-lm` / MLX runtime on Apple Silicon | [`results/apple_m3_max_native_performance.csv`](results/apple_m3_max_native_performance.csv) |

## Native MLX performance

These measurements use the `M` checkpoint on an Apple M3 Max with 69 GB of
unified memory. Each run uses 1,024 input tokens, 1,024 generated tokens,
256-token chunked prefill, and the best result from five repetitions.

`TTFT` includes prefill and the first generated token. Decode throughput is the
steady-state token rate. Peak memory is reported by the MLX Metal allocator.
The recorded run was added in
[`d5de613`](https://github.com/TheStageAI/edge-lm/commit/d5de613427af865cfd7fe2a4ca574f3445911734).

### Gemma 4 E2B

| Model | TTFT | Decode | MLX peak memory |
| --- | ---: | ---: | ---: |
| **TheStage M** | **434 ms** | **115.0 tok/s** | **2.1 GB** |
| Reference BF16 | 531 ms | 57.2 tok/s | 10.7 GB |
| Reference 4-bit, group size 32 | 595 ms | 83.3 tok/s | 4.6 GB |

### Gemma 4 E4B

| Model | TTFT | Decode | MLX peak memory |
| --- | ---: | ---: | ---: |
| **TheStage M** | **832 ms** | **73.7 tok/s** | **3.5 GB** |
| Reference BF16 | 1,110 ms | 30.5 tok/s | 16.4 GB |
| Reference 4-bit, group size 32 | 970 ms | 53.5 tok/s | 7.1 GB |

Reproduce the native comparison:

```bash
python benchmarks/performance.py \
  --model TheStageAI/gemma-4-E2B-it \
  --hf-model google/gemma-4-E2B-it \
  --input-tokens 1024 \
  --output-tokens 1024 \
  --prefill-step-size 256 \
  --compare-ref \
  --compare-ref-4bit \
  --ref-4bit-group-size 32
```

## Quality benchmarks

The quality scripts materialize native MLX checkpoints as standard Hugging Face
weights. They dequantize GGUF comparison baselines into the same layout, then
serve every checkpoint through vLLM.

The detailed protocol, complete tables, frozen configs, and end-to-end
verification commands are in [`quality/`](quality/).

## Portable GGUF releases

The Qwen 3.5 and Gemma 4 GGUF releases keep their manifests and benchmark
tables with the artifacts on Hugging Face. The
[TheStageAI Edge Models collection](https://huggingface.co/collections/TheStageAI/edge-lm-6a5f36315458277fa7b9e26a)
links the portable release line alongside the native models.

GGUF rows inside [`quality/`](quality/) are comparison baselines for the native
Gemma 4 releases. They do not measure `edge-lm` runtime performance.

## Reporting conventions

- Artifact size is storage size, not total runtime memory.
- IFEval `P / I` means prompt-strict / instruction-strict accuracy.
- Quality and performance keep their original execution paths.
- Each result records its hardware, context length, sampling settings, and
  protocol revision.
