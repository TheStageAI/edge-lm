# Benchmarks

Benchmarks are grouped by model family and execution contract. Native Gemma 4
quality uses an equalized vLLM path, native performance runs in MLX, and the
portable Qwen3.5 and Gemma 4 releases keep their own frozen result snapshots.

| Track | Question | Execution path | Canonical results |
| --- | --- | --- | --- |
| [Native Gemma 4 quality](quality/) | How much behavior survives native compression? | Backend-equalized Hugging Face checkpoints served through vLLM | [`quality/results/gemma4_release_quality.csv`](quality/results/gemma4_release_quality.csv) |
| [Native MLX performance](#native-mlx-performance) | How fast and memory-efficient is the deployed model? | Native `edge-lm` / MLX runtime on Apple Silicon | [`results/apple_m3_max_native_performance.csv`](results/apple_m3_max_native_performance.csv) |
| [Qwen3.5 GGUF](qwen35-GGUF/) | Which portable tier fits each Qwen3.5 model? | Release-aligned Hugging Face mirrors served through vLLM | [`qwen35-GGUF/quality.csv`](qwen35-GGUF/quality.csv) |
| [Gemma 4 GGUF](gemma4-GGUF/) | Which portable tier fits each Gemma 4 model? | Release-aligned Hugging Face mirrors served through vLLM | [`gemma4-GGUF/quality.csv`](gemma4-GGUF/quality.csv) |

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

## Native Gemma 4 quality

The quality scripts materialize native MLX checkpoints as standard Hugging Face
weights. They dequantize GGUF comparison baselines into the same layout, then
serve every checkpoint through vLLM.

The detailed protocol, complete tables, frozen configs, and end-to-end
verification commands are in [`quality/`](quality/).

## Portable GGUF quality

The family snapshots keep the public release tables close to the repository
that introduces them:

- [`qwen35-GGUF/`](qwen35-GGUF/) covers four models, 16 shipping files, 20
  BF16/XS/S/M/L quality rows, and the Qwen long-thinking reporting boundary.
- [`gemma4-GGUF/`](gemma4-GGUF/) covers three models, 12 shipping files, and 15
  complete BF16/XS/S/M/L quality rows.

Each directory contains a readable summary plus separate artifact and quality
CSVs. The artifact tables preserve exact filenames, bytes, BPW, KL, hashes,
base revisions, and evaluation IDs. The
[TheStageAI Edge Models collection](https://huggingface.co/collections/TheStageAI/edge-lm-6a5f36315458277fa7b9e26a)
links the native and portable model repositories.

GGUF rows inside [`quality/`](quality/) are comparison baselines for the native
Gemma 4 releases. They do not measure `edge-lm` runtime performance.

## Reporting conventions

- Artifact size is storage size, not total runtime memory.
- IFEval `P / I` means prompt-strict / instruction-strict accuracy.
- Quality and performance keep their original execution paths.
- Missing MMLU-Pro values are marked `not_reported`; partial subject runs are
  not promoted to model-level scores.
- Results from different execution contracts stay in separate tables.
