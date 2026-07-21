# Qwen3.5 GGUF benchmarks

[← Benchmark index](../)

This directory is the public benchmark snapshot for the four Qwen3.5 GGUF
repositories. Each repository contains XS, S, M, and L checkpoints for
llama.cpp-compatible runtimes.

## Recommended checkpoints

| Model | Start with | Size | IFEval P / I (%) | MMLU-Pro (%) |
| --- | ---: | ---: | ---: | ---: |
| [Qwen3.5 0.8B](https://huggingface.co/TheStageAI/Qwen3.5-0.8B-GGUF) | **S** | 418 MB | 50.65 / 62.11 | — |
| [Qwen3.5 2B](https://huggingface.co/TheStageAI/Qwen3.5-2B-GGUF) | **M** | 1.07 GB | 66.54 / 75.54 | — |
| [Qwen3.5 4B](https://huggingface.co/TheStageAI/Qwen3.5-4B-GGUF) | **M** | 2.39 GB | 80.22 / 86.09 | 78.86 |
| [Qwen3.5 9B](https://huggingface.co/TheStageAI/Qwen3.5-9B-GGUF) | **M** | 5.06 GB | 83.36 / 88.49 | 82.16 |

Each recommendation follows the best measured size and quality trade-off for
that model.

## Evaluation modes

- **IFEval:** 541 prompts with the native chat template, thinking disabled,
  temperature 0, and a 1,280-token generation limit.
- **MMLU-Pro:** 12,032 questions with the native chat template, thinking
  enabled, temperature 1, top-p 0.95, and a 32,768-token generation limit.

These modes measure different behavior. XS remains useful for compact
non-thinking instruction following, but its long-thinking trajectories are
less stable. Complete MMLU-Pro scores are reported for Qwen3.5 4B and 9B at
BF16, S, M, and L. Other cells remain unreported; partial subject runs are not
converted into model-level scores.

Downstream quality runs use release-aligned Hugging Face mirrors served through
vLLM. The shipping GGUFs pass separate export and llama.cpp load gates. These
tables compare model behavior; they are not llama.cpp performance benchmarks.

## Complete quality results

Scores are percentages. P / I means prompt-strict / instruction-strict IFEval.

### Qwen3.5 0.8B

| Variant | IFEval P / I | MMLU-Pro |
| --- | ---: | ---: |
| BF16 | 52.13 / 63.79 | — |
| XS | 38.08 / 50.24 | — |
| **S** | **50.65 / 62.11** | — |
| M | 49.17 / 60.43 | — |
| L | 53.79 / 63.91 | — |

### Qwen3.5 2B

| Variant | IFEval P / I | MMLU-Pro |
| --- | ---: | ---: |
| BF16 | 65.43 / 74.70 | — |
| XS | 52.68 / 64.15 | — |
| S | 63.22 / 73.38 | — |
| **M** | **66.54 / 75.54** | — |
| L | 65.80 / 74.94 | — |

### Qwen3.5 4B

| Variant | IFEval P / I | MMLU-Pro |
| --- | ---: | ---: |
| BF16 | 82.44 / 87.53 | 79.55 |
| XS | 70.43 / 78.30 | — |
| S | 77.82 / 83.93 | 74.39 |
| **M** | **80.22 / 86.09** | **78.86** |
| L | 81.70 / 87.05 | 79.59 |

### Qwen3.5 9B

| Variant | IFEval P / I | MMLU-Pro |
| --- | ---: | ---: |
| BF16 | 83.18 / 88.37 | 82.39 |
| XS | 79.30 / 85.13 | — |
| S | 82.81 / 87.65 | 79.70 |
| **M** | **83.36 / 88.49** | **82.16** |
| L | 82.99 / 87.89 | 82.40 |

## Machine-readable results

- [`quality.csv`](quality.csv) contains all 20 BF16/XS/S/M/L evaluation rows.
- [`artifacts.csv`](artifacts.csv) contains the 16 shipping GGUFs with exact
  filenames, byte sizes, whole-file BPW, file types, held-out KL, revisions,
  evaluation IDs, and SHA-256 digests.

Scores in `quality.csv` are stored as fractions. A blank MMLU-Pro value paired
with `not_reported` means that no complete model-level score is published.

## Snapshot provenance

The tables were derived from the verified 28-checkpoint release ledger and the
complete release benchmark aggregate. Source snapshot hashes:

- benchmark matrix: `3ca0546eff55ee761e54dcecc9eec366c41f6d62e96b00dc7c22cad067dd4233`
- checkpoint manifest: `c03954d8d3751a7b1402945b1fc1ed1aa38683a8220da87e0cc312449daf5dad`
- artifact metadata: `ced7b42d19c2eed3b06ea0b43e54b2e3a3303183ad11ebd69e611f763b342495`
