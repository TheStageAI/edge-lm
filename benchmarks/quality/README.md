# Production Quality Benchmarks

These files document the production protocol behind the release quality tables.
The headline numbers do not come from the MLX runtime. For quality comparisons,
release checkpoints are evaluated through the same vLLM-backed benchmark path.

TheStage MLX release checkpoints are materialized as standard Hugging Face BF16
checkpoints before vLLM evaluation. Public GGUF baselines are downloaded from
Hugging Face, dequantized with the `gguf` reader into the same Hugging Face BF16
key layout, and then served through vLLM. The MLX scripts remain useful for
local runtime checks, but they are not the source of the headline quality table.

## Protocols

- `protocols/gemma4_mmlu_pro_vllm.json`: official MMLU-Pro release protocol.
  It uses `TIGER-Lab/MMLU-Pro`, vLLM, Gemma 4 chat formatting, 0-shot
  chain-of-thought prompting, thinking enabled, and one shard per subject.
- `protocols/gemma4_ifeval_vllm.json`: IFEval release protocol through the
  vLLM/lm-eval path with chat templates and deterministic generation.
- `protocols/gemma4_tau2_vllm_qwen_user.json`: Tau2 release protocol. The model
  under test is served as the agent through vLLM; the simulated user is fixed to
  `Qwen3-235B-A22B-2507`.
- `release_models.json`: public HF source artifacts for the release comparison
  table, including TheStage MLX checkpoints and Unsloth GGUF baselines.

## Results

`IFEval p/i` means prompt strict / instruction strict. The IFEval numbers below
use the corrected public recipe with `max_gen_toks=1280`.

### Original Gemma 4 Release

Every model in this table is materialized to the same standard BF16 evaluation
path and served through vLLM. Tau2 uses `Qwen3-235B-A22B-2507` as the user
simulator.

**Gemma 4 E2B**

| Model | Compression | MMLU-Pro | IFEval p/i | Tau2 |
|---|---:|---:|---:|---:|
| BF16 | 1.00x | 61.85 | 75.23 / 82.37 | 30.67 |
| TheStage L | 5.62x | 54.48 | 76.34 / 83.45 | 22.20 |
| TheStage M | 6.40x | 49.85 | 75.23 / 82.61 | 23.45 |
| Unsloth Q3-K-S | 3.81x | 48.20 | 66.36 / 76.02 | 18.69 |
| Unsloth UD-Q2-K-XL | 3.87x | 43.17 | 66.54 / 76.38 | 20.23 |

**Gemma 4 E4B**

| Model | Compression | MMLU-Pro | IFEval p/i | Tau2 |
|---|---:|---:|---:|---:|
| BF16 | 1.00x | 70.49 | 85.03 / 89.57 | 37.19 |
| TheStage L | 4.64x | 67.41 | 84.66 / 89.33 | 33.25 |
| TheStage M | 5.60x | 63.54 | 81.33 / 87.05 | 29.04 |
| Unsloth Q3-K-S | 3.90x | 63.66 | 81.15 / 87.17 | 30.47 |
| Unsloth UD-Q2-K-XL | 4.01x | 58.69 | 82.81 / 88.25 | 22.91 |

### QAT-Source Release

These rows use Google's QAT-trained BF16 checkpoints as the compression source.
Native rows are `edge-lm` checkpoints. GGUF rows are portable llama.cpp
artifacts, evaluated through the same dequantized BF16 path for quality.

**Gemma 4 E2B**

| Model | Size | MMLU-Pro | IFEval p/i |
|---|---:|---:|---:|
| BF16 reference | 10.21 GB | 61.85 | 75.23 / 82.37 |
| QAT BF16 dequantized | 10.21 GB | 59.30 | 72.46 / 80.70 |
| TheStage native M | 1.44 GB | 47.91 | 75.42 / 83.09 |
| TheStage native L | 1.72 GB | 54.45 | 76.71 / 83.69 |
| TheStage GGUF M | 2.47 GB | 53.79 | 72.64 / 81.29 |
| TheStage GGUF L | 2.68 GB | 57.12 | 73.38 / 81.65 |
| TheStage GGUF W4-uniform | 2.69 GB | 56.91 | 74.68 / 82.61 |

**Gemma 4 E4B**

| Model | Size | MMLU-Pro | IFEval p/i |
|---|---:|---:|---:|
| BF16 reference | 15.88 GB | 70.49 | 85.03 / 89.57 |
| QAT BF16 dequantized | 15.88 GB | 69.08 | 78.56 / 84.41 |
| TheStage native M | 2.72 GB | 63.67 | 81.70 / 87.17 |
| TheStage native L | 3.27 GB | 67.53 | 85.40 / 89.69 |
| TheStage GGUF M | 4.08 GB | 65.44 | 81.33 / 87.29 |
| TheStage GGUF L | 4.40 GB | 67.90 | 82.81 / 87.89 |
| TheStage GGUF W4-uniform | 4.40 GB | 68.34 | 81.33 / 87.05 |

The machine-readable copy of the headline table is
[`results/gemma4_release_quality.csv`](results/gemma4_release_quality.csv).

## End-to-End Verification

Use `verify_release.py` to download public HF artifacts, materialize them into
the format expected by the vLLM backend, and run the production eval path.

Example: compare our E2B `M` checkpoint against the Unsloth Q3-K-S baseline on
IFEval:

```bash
python benchmarks/quality/verify_release.py \
  --work-dir runs/release_verify \
  run \
  --models e2b_ours_m,e2b_unsloth_q3_k_s \
  --benchmarks ifeval
```

Run one MMLU-Pro subject shard:

```bash
python benchmarks/quality/verify_release.py \
  --work-dir runs/release_verify \
  run \
  --models e2b_ours_m,e2b_unsloth_q3_k_s \
  --benchmarks mmlu_pro \
  --subjects biology
```

Run the full production set by omitting `--subjects`:

```bash
python benchmarks/quality/verify_release.py \
  --work-dir runs/release_verify \
  run \
  --models e2b_ours_m,e2b_unsloth_q3_k_s \
  --benchmarks mmlu_pro,ifeval
```

Materialization details:

- TheStage checkpoints are downloaded from their public HF repos as MLX
  `model_{m,l}.safetensors`, compact `ple_{m,l}.safetensors`, and shared
  audio/vision tower files, then dequantized into standard Hugging Face
  safetensors. Missing tokenizer/processor metadata is copied from the public
  base Gemma checkpoint so vLLM can load the materialized directory directly.
- Unsloth baselines are downloaded as GGUF files, dequantized with the `gguf`
  reader, and saved as standard Hugging Face safetensors with tokenizer/config
  metadata copied from the public base Gemma checkpoint. For text-only
  benchmarks, the materialized GGUF config uses vLLM's `Gemma4ForCausalLM`
  architecture so audio/vision tower weights are not required.

## MMLU-Pro

Run one subject shard:

```bash
python benchmarks/quality/mmlu_pro_vllm.py \
  --protocol benchmarks/quality/protocols/gemma4_mmlu_pro_vllm.json \
  run-subject \
  --model /path/to/hf_bf16_checkpoint \
  --subject biology \
  --output-dir runs/mmlu_pro/<run_id>/subjects
```

Run all subjects by launching one `run-subject` command per subject listed in
the protocol. The production setup runs these shards in parallel on H100s.

Aggregate subject outputs:

```bash
python benchmarks/quality/mmlu_pro_vllm.py \
  --protocol benchmarks/quality/protocols/gemma4_mmlu_pro_vllm.json \
  aggregate \
  --input-dir runs/mmlu_pro/<run_id>/subjects \
  --output runs/mmlu_pro/<run_id>/summary.official_random.json
```

Each subject shard writes `<subject>.json` with raw model outputs and
`<subject>.summary.json` with shard diagnostics. The aggregate writes
`summary.official_random.json`.

## IFEval

IFEval is run through the same standard-HF-checkpoint and vLLM backend family.
The frozen settings are in `protocols/gemma4_ifeval_vllm.json`. The important
release choices are:

- `tasks=["ifeval"]`
- `apply_chat_template=true`
- `fewshot_as_multiturn=true`
- `enable_thinking=false`
- deterministic generation
- stop strings: `<end_of_turn>`, `<turn|>`, `<eos>`
- `max_gen_toks=1280`

Run IFEval:

```bash
python benchmarks/quality/lm_eval_vllm.py \
  --protocol benchmarks/quality/protocols/gemma4_ifeval_vllm.json \
  --model /path/to/hf_bf16_checkpoint \
  --output-dir runs/ifeval/<run_id>
```

## Tau2

Tau2 uses the checkpoint under test as the tool-using agent and keeps the user
simulator fixed. The release protocol evaluates the full task set across
`airline`, `retail`, and `telecom`.

The frozen settings are in `protocols/gemma4_tau2_vllm_qwen_user.json`.
