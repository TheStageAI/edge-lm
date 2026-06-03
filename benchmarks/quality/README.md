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
- `max_gen_toks=768`

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

## Results

`results/gemma4_release_quality.csv` contains the headline table values in a
machine-readable format. README tables should be generated or checked against
this CSV before release.
