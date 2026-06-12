# edge-lm

![Gemma E2B compression flow: 9.26 GB BF16 compressed to 1.44 GB — 6.4× smaller](https://cdn.thestage.ai/production/cms_file_upload/1780406294-645b80f9-cebe-4ef2-bc04-f524afb4f244/Tokens%20per%20Second%20CuDNN%20%282%29.png)

**Tiny LLMs optimized for edge deployment.**

`edge-lm` runs compressed large language models on-device — Apple Silicon Macs and iPhones — through [MLX](https://github.com/ml-explore/mlx). The first release ships the **smallest publicly available Gemma 4 checkpoints optimized for edge deployment** — roughly **7× smaller** than the original while preserving the capabilities that matter most for on-device assistants: general world knowledge, instruction following, and tool use.


> 📝 Read the full write-up: [*7× size reduction for Gemma 4 Edge models — Compressing PLE architectures*](https://app.thestage.ai/blog/7x-size-reduction-for-Gemma4-Edge-models?id=14).

## Models

| Model | Source | M size (default) | L size | Compression | GGUF / llama.cpp |
|---|---|---:|---:|---:|---|
| [`TheStageAI/gemma-4-E2B-it`](https://huggingface.co/TheStageAI/gemma-4-E2B-it) | `google/gemma-4-E2B-it` | **1.44 GB** | 1.72 GB | up to 6.4× | n/a |
| [`TheStageAI/gemma-4-E4B-it`](https://huggingface.co/TheStageAI/gemma-4-E4B-it) | `google/gemma-4-E4B-it` | **2.72 GB** | 3.28 GB | up to 5.6× | n/a |
| [`TheStageAI/gemma-4-E2B-it-qat`](https://huggingface.co/TheStageAI/gemma-4-E2B-it-qat) | `google/gemma-4-E2B-it-qat-q4_0-unquantized` | **1.44 GB** | 1.72 GB | up to 6.4× | [`GGUF`](https://huggingface.co/TheStageAI/gemma-4-E2B-it-qat-GGUF) |
| [`TheStageAI/gemma-4-E4B-it-qat`](https://huggingface.co/TheStageAI/gemma-4-E4B-it-qat) | `google/gemma-4-E4B-it-qat-q4_0-unquantized` | **2.72 GB** | 3.27 GB | up to 5.6× | [`GGUF`](https://huggingface.co/TheStageAI/gemma-4-E4B-it-qat-GGUF) |

Weights download automatically from HuggingFace on first run. Each model ships two operating points — `l` (more quality, larger artifact) and `m` (the smaller headline compression target, default).
The GGUF links are provided for llama.cpp-compatible deployment. They are not native `edge-lm` checkpoints and are not loaded by this library.

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

![Gemma 4 E4B Pareto: size vs quality](benchmarks/quality/assets/gemma4_e4b_pareto_size_quality.png)

Full quality tables, evaluation settings, and reproduction commands are in
[`benchmarks/quality`](benchmarks/quality/).

## License

Released under the [MIT License](LICENSE), © 2026 thestage.ai labs.

The compressed model weights are derivatives of Google's Gemma 4 and are additionally subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
