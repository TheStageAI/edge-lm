# edge-lm

![Gemma E2B compression flow: 9.26 GB BF16 compressed to 1.44 GB — 6.4× smaller](https://cdn.thestage.ai/production/cms_file_upload/1780406294-645b80f9-cebe-4ef2-bc04-f524afb4f244/Tokens%20per%20Second%20CuDNN%20%282%29.png)

**Tiny LLMs optimized for edge deployment.**

`edge-lm` runs compressed large language models on-device — Apple Silicon Macs and iPhones — through [MLX](https://github.com/ml-explore/mlx). The first release ships the **smallest publicly available Gemma 4 checkpoints optimized for edge deployment** — roughly **7× smaller** than the original while preserving the capabilities that matter most for on-device assistants: general world knowledge, instruction following, and tool use.


> 📝 Read the full write-up: [*7× size reduction for Gemma 4 Edge models — Compressing PLE architectures*](https://app.thestage.ai/blog/7x-size-reduction-for-Gemma4-Edge-models?id=14).

## Models

| Model | M size (default) | L size | Compression |
|---|---|---|---|
| [`TheStageAI/gemma-4-E2B-it`](https://huggingface.co/TheStageAI/gemma-4-E2B-it) | **1.44 GB** | 1.72 GB | up to 6.4× |
| [`TheStageAI/gemma-4-E4B-it`](https://huggingface.co/TheStageAI/gemma-4-E4B-it) | **2.72 GB** | 3.28 GB | up to 5.6× |

Weights download automatically from HuggingFace on first run. Each model ships two operating points — `l` (more quality, larger artifact) and `m` (the smaller headline compression target, default).

## Key features

- **~7× smaller checkpoints.** The default Gemma 4 E2B checkpoint fits in 1.44 GB, and E4B fits in 2.72 GB — small enough to download quickly and stay within mobile per-app memory budgets.
- **Accuracy preserved where it counts.** Quality is held on the three things that matter most for edge assistants — instruction following (IFEval), tool calls (τ²-Bench), and general world knowledge (MMLU-Pro).
- **MLX-ready artifacts.** Decoder weights use a flat, MLX-compatible per-group quantization format; PLE tables use a compact AQLM-style vector-quantization codec (4.7 GB → ~0.26 GB), decompressed on the fly with a single batched gather.

## Quick start

```bash
git clone https://github.com/TheStageAI/edge-lm.git
cd edge-lm

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # or: pip install -e .
```

Run text generation (downloads `TheStageAI/gemma-4-E2B-it` on first run):

```bash
python examples/generation_test.py --prompts "What is 2+2?" "Explain gravity in one sentence"
```

Use it from Python:

```python
from edge_lm import load
from mlx_vlm import stream_generate

model, tokenizer = load()  # TheStageAI/gemma-4-E2B-it, size "m" by default
# model, tokenizer = load("TheStageAI/gemma-4-E4B-it", size="l")  # larger, higher quality

prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Write a haiku about the moon."}],
    tokenize=False, add_generation_prompt=True,
)
for chunk in stream_generate(model, tokenizer, prompt, max_tokens=128):
    print(chunk.text, end="", flush=True)
```

More examples:

```bash
python examples/test_vision.py --image photo.jpg --prompt "Describe this image"
python examples/test_audio.py  --audio recording.wav --prompt "Transcribe this speech"
python examples/chat.py --tools                      # interactive chat with tool use
```

## Benchmarks

### Quality

Every model — ours and the GGUF baselines alike — is dequantized to a standard BF16 checkpoint and served through vLLM, so the backend is equalized across the table. We report **MMLU-Pro** (general knowledge), **IFEval** (instruction following), and **τ²-Bench / Tau2** (multi-step tool use). For Tau2 the Gemma checkpoint under test acts as the agent while a fixed `Qwen3-235B-A22B-2507` simulates the user.

`Ours L` keeps more quality at a larger artifact size; `Ours M` is the smaller headline compression target.

**Gemma 4 E2B**

| Model | Compression | MMLU-Pro | IFEval | Tau2 (avg of 3) |
|---|---|---|---|---|
| BF16 | 1.00× | 61.85 | 74.68 | 30.67 |
| **Ours L** | 5.62× | **54.48** | **74.86** | 22.20 |
| **Ours M** | **6.40×** | 49.85 | 71.53 | **23.45** |
| Unsloth Q3-K-S | 3.81× | 48.20 | 64.51 | 18.69 |
| Unsloth UD-Q2-K-XL | 3.87× | 43.17 | 66.54 | 20.23 |

**Gemma 4 E4B**

| Model | Compression | MMLU-Pro | IFEval | Tau2 |
|---|---|---|---|---|
| BF16 | 1.00× | 70.49 | 81.33 | 37.19 |
| **Ours L** | 4.64× | **67.41** | **81.52** | **33.25** |
| **Ours M** | **5.60×** | 63.54 | 80.78 | 29.04 |
| Unsloth Q3-K-S | 3.90× | 63.66 | 77.08 | 30.47 |
| Unsloth UD-Q2-K-XL | 4.01× | 58.69 | 79.67 | 22.91 |

Bold metric values mark the best result among the compressed checkpoints in each column. Tau2 computed with `Qwen3-235B-A22B-2507` as the user simulator.

Reproduce the quality benchmarks:

```bash
pip install "edge-lm[quality]"   # CUDA/vLLM quality benchmark dependencies

python benchmarks/quality/verify_release.py \
    --work-dir runs/release_verify \
    run \
    --models e2b_ours_m,e2b_unsloth_q3_k_s \
    --benchmarks mmlu_pro,ifeval
```

The frozen production protocols live in [`benchmarks/quality`](benchmarks/quality/).

### Performance

Measured on an **Apple M3 Max (69 GB)**, size `m` checkpoint, 1024 input / 1024 output tokens,
chunked prefill (256-token chunks), best of 5 runs. `TTFT` = prefill + first token;
`TPS` = steady-state decode throughput; `MLX peak memory` = `mx.get_peak_memory()` (MLX Metal allocator).
References are the matching original `google/gemma-4-*-it` checkpoint served via mlx-vlm: bf16,
and 4-bit quantized (affine, group size 32).

**Gemma 4 E2B**

| Model | TTFT | Decode (TPS) | MLX peak memory |
|---|---|---|---|
| **TheStage (ours)** | **434 ms** | **115.0** | **2.1 GB** |
| Reference bf16 | 531 ms | 57.2 | 10.7 GB |
| Reference 4-bit (gs32) | 595 ms | 83.3 | 4.6 GB |

**Gemma 4 E4B**

| Model | TTFT | Decode (TPS) | MLX peak memory |
|---|---|---|---|
| **TheStage (ours)** | **832 ms** | **73.7** | **3.5 GB** |
| Reference bf16 | 1110 ms | 30.5 | 16.4 GB |
| Reference 4-bit (gs32) | 970 ms | 53.5 | 7.1 GB |

Reproduce:

```bash
python benchmarks/performance.py --model TheStageAI/gemma-4-E2B-it \
    --hf-model google/gemma-4-E2B-it \
    --input-tokens 1024 --output-tokens 1024 --prefill-step-size 256 \
    --compare-ref --compare-ref-4bit --ref-4bit-group-size 32
```

## License

Released under the [MIT License](LICENSE), © 2026 thestage.ai labs.

The compressed model weights are derivatives of Google's Gemma 4 and are additionally subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
