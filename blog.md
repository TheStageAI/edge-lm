## Abstract

We present a series of compressed Gemma-4 checkpoints that are roughly 7× smaller than the original while preserving the capabilities that matter most for on-device assistants: general world knowledge, instruction following, and tool use. We achieve this by compressing the model along its natural structure—non-uniform, mixed-precision quantization of the transformer backbone, together with up-to-20× vector quantization of the Per-Layer Embedding (PLE) tables, which hold nearly half of the parameters. The optimized checkpoints are packaged for Apple Silicon Macs and iPhones via MLX, fit within mobile memory budgets, and target practical deployment on phones, laptops, and other edge devices.

<p align="center">
  <img src="https://cdn.thestage.ai/production/cms_file_upload/1780411906-d0180e32-8236-45f7-a0a1-d216a4dcb735/pareto-3.png" alt="Pareto for IFEval" width="48%" style="display: inline-block; margin-right: 1%;">
  <img src="https://cdn.thestage.ai/production/cms_file_upload/1780411851-c8e45f93-1e45-471f-85db-87f7a3881f6c/pareto_mmlu-2.png" alt="Pareto for MMLU" width="48%" style="display: inline-block;">
</p>

## Introduction

Large language models (LLMs) have become markedly more capable in recent years. Their *intelligence density*—capability per parameter—has risen sharply: recent models follow instructions, call tools, and solve hard problems far better than their predecessors while using fewer parameters. Releases such as Gemma-4 and the Qwen-3.5/3.6 family deliver strong reasoning and agentic behavior at only a few billion parameters, putting capable models, in principle, within reach of laptops and smartphones.

In practice, three properties still block edge deployment: checkpoint size on disk, runtime memory footprint, and generation speed. All three are amplified on mobile hardware, where resources are scarce and tightly governed by the operating system.

iOS makes this concrete. The system enforces a per-application memory budget—roughly 3 GB on many devices—and terminates processes that exceed it. A standard Gemma checkpoint either consumes nearly all of an app's memory or forces unused weights to be offloaded to disk and streamed back on demand, putting I/O on the critical path and capping generation speed. Size also bites before loading: few users will wait for a 3–5 GB download to use a single feature.

Quantization can address all three constraints at once—smaller files, smaller footprint, and faster inference—but it comes with quality degradation, especially when the model is quantized below 4 bits.

We close this gap with a series of Gemma-4 checkpoints built for edge deployment. We compress the transformer backbone with a combination of state-of-the-art non-uniform post-training quantization techniques, and exploit a structural property of the Per-Layer Embedding (PLE) architecture to compress the embeddings far more aggressively with vector quantization. 

Our contributions are:

- **~7× smaller.** The checkpoints are roughly 7× smaller than the original Gemma-4. TheStageAI/gemma-4-E2B-it fits to 1.4 gb and TheStageAI/gemma-4-E4B-it fits to 2.6 gb.
- **Accuracy preserved where it counts.** They hold their accuracy on the three things that matter most for edge LLMs—instruction following, tool calls, and general world knowledge.
- **MLX-ready artifacts.** They use an MLX-compatible plain per-group format for the Apple Silicon release path.

Release model cards: TheStageAI/gemma-4-E2B-it, TheStageAI/gemma-4-E4B-it.

# Related works

---

The strongest open on-device baselines mostly come from the GGUF/llama.cpp ecosystem, including Unsloth releases. These checkpoints are not just low-bit weights: they rely on runtime-specific layouts such as k-quants/i-quants, compact block metadata, and kernels tuned around that format. This makes them excellent public baselines, but not drop-in artifacts for our MLX release path, where we want plain per-group weight-only tensors.

For transformer weights, the closest methodological baseline is GPTQ, which uses approximate second-order information to minimize local reconstruction error after quantization. AWQ is another strong weight-only PTQ family, protecting activation-salient weights during low-bit compression. SmoothQuant attacks a different problem: it smooths activation outliers by redistributing scale between activations and weights. That is less aligned with our goal, because we want to preserve a simple weight-only inference graph without activation-side transforms or runtime graph changes.

Rotation-based methods such as SpinQuant and QuIP-style quantization improve low-bit PTQ by changing the basis in which weights are quantized, using hidden gauge symmetries to insert rotations that later cancel out. Gemma 4 is a poor fit for making this the main path: its PLE-heavy design, tied/shared embedding structure, and MLX runtime target leave only a small useful gauge group, so there is little freedom to rotate the model while preserving the deployed graph.

Vector and additive quantization methods, including product quantization and AQLM, can be much stronger than scalar quantization when a tensor is naturally represented through learned codebooks. We use that idea only for PLE, where codebook lookup is a good match for the storage bottleneck. For transformer linear weights, we deliberately avoid vector-quantized kernels and keep ordinary per-group quantization, so inference remains compatible with standard MLX weight-only execution.

Finally, mixed-bit compression needs a rule for where the bits go. A common approach is to rank layers by local sensitivity and assign more precision to the layers that look fragile. GGUF importance matrices (`imatrix`) are a stronger practical version of this idea: calibration data is used to weight quantization error by importance rather than treating all coordinates equally. Riemannian Constrained Optimization (RCO) goes one step further for our setting: it searches over full compression choices under an exact byte budget and optimizes the model-level KL objective directly, rather than relying only on local sensitivity scores.


# PLE Transformers Compression

---

![Tokens per Second CuDNN](https://cdn.thestage.ai/production/cms_file_upload/1780406294-645b80f9-cebe-4ef2-bc04-f524afb4f244/Tokens%20per%20Second%20CuDNN%20(2).png)

## Architecture overview

---

Gemma 4 E2B/E4B build on the Gemma 3n design lineage, refining it in several ways. Their defining features include a hybrid attention mechanism that interleaves local sliding-window and global layers, per-layer embeddings (PLE), and tied LM-head and token embeddings for memory-efficient long-context inference. The global layers share unified keys and values and apply Proportional RoPE (p-RoPE). Notably, the "2B"/"4B" designations refer to *effective* parameters — the actual checkpoints are considerably larger (5.1B/8B) due to the PLE parameters.

## Pipeline Overview

---

We compress Gemma 4 along the parts of the model that dominate either size or quality, then choose the final precision schedule under a fixed artifact budget.

- **Transformer blocks:** decoder projections are compressed with GPTQ, Quantization Error Propagation (QEP), and range clipping into MLX-compatible plain per-group weight-only tensors.
- **PLE tables:** per-layer token embeddings are compressed with an AQLM-style vector-quantization codec, using sensitivity-weighted assignments instead of ordinary scalar codes.
- **Token embeddings / LM head:** the shared embedding path is handled separately with a simple scalar quantization scheme matched to the same runtime constraints.
- **Schedule search:** Riemannian Constrained Optimization (RCO) selects the bit-width and group-size option for each transformer module from a bank of already-quantized candidates, while keeping the total byte budget fixed.
- **Final release checkpoint:** we do not ship the RCO bank splice directly. The learned schedule is used to re-quantize the dense model in one consistent GPTQ/QEP pass, then combined with the PLE codec for the release artifact.

## Compressing transformer

---

The transformer stack is where we keep the runtime format deliberately simple. Our release path targets MLX on-device kernels, so each decoder projection is emitted as plain weight-only per-group quantization: asymmetric integer weights with one scale and zero point per 32/64/128-channel group. This is different from the strongest community GGUF/llama.cpp checkpoints, which often use hierarchical k-quant superblocks and smaller effective groups. Those checkpoints are useful public baselines and can be adapted to other runtimes, but they are not the format we ship for MLX.

The quality of a low-bit GPTQ checkpoint depends heavily on the activations used during calibration. Since our target is an on-device assistant, we calibrate on a Gemma-specific, chat-shaped mix rather than a generic Wikitext-style corpus.

The production GPTQ pass reads deterministic row views: each JSONL row is one fixed 5120-token calibration window, so source balancing and document boundaries are decided before the quantizer runs. The current view contains 512 windows drawn from self-calibration generations, selected multi-turn traces, synthetic instruction and domain prompts, and small public slices for reasoning, safety, and general coverage. This gives the Hessian and QEP estimates activations closer to the instruction-following, tool-use, and general-assistant workloads we care about on device.

This calibration set is not used as the public benchmark. RCO schedule search uses a separate chat-template train/validation split, and final quality is checked on held-out generated distribution-proxy prompts plus MMLU-Pro, IFEval, and Tau2.

The quantizer is GPTQ [Frantar et al., 2022], run sequentially down the stack: once a layer is quantized, calibration activations are pushed through it before moving on, so every later layer is calibrated against the drifted activations the deployed model will really produce, not an idealized dense path. We point readers to the paper for the mechanics; what matters here are the two additions that make a flat low-bit format work.

Range clipping (`mseclip`) keeps a few outliers from forcing a whole group to waste its code range on rare values — before quantizing, we search shrink factors and keep the interval that minimizes reconstruction error. 

Quantization Error Propagation (QEP) handles cross-module drift: since each module receives inputs already shifted by earlier quantization, QEP measures that drift during calibration and pre-shifts the dense weight to match the activations the module will actually see. Both are calibration-time only — no runtime cost, no extra parameters, folded into the weights before the integer codes are emitted. The runtime just loads plain per-group weights.

### QEP ablation

To isolate Quantization Error Propagation, we compare GPTQ and GPTQ+QEP at the same transformer quantization setting, with the same calibration data and range clipping. The metric is held-out distribution-proxy KL; lower is better.

| Method | E2B | E4B |
| --- | --- | --- |
| GPTQ |  0.5667  | 0.2378 |
| GPTQ+QEP | 0.5278 | 0.2256 |

QEP improves distribution matching under the same storage format, so we keep it enabled in all release checkpoints.

## Compressing embeddings

---

Gemma 4 E2B/E4B could be described as a 2B/4B model. The dense checkpoints are closer to 5.1B and 8B parameters once embeddings are counted. The gap is per-layer embeddings (PLE): a table that assigns every vocabulary token a distinct vector at each depth of the network. PLE is much of what lets the E-series perform well above its effective parameter count, and it is also what inflates the on-device footprint.

Conceptually, it is a `token × PLE-layer × channel tensor`: for every vocabulary token, the model stores a separate PLE vector at every depth.
In BF16, this table alone takes about 4.7 GB; even a direct Q4 representation would still be about 1.17 GB. Our final packed representation brings it down to about 0.26 GB, including both indices and codebooks.

The reason this needed a separate codec is that PLE is not just another smooth embedding table. Empirically, it behaves more like dense layer-specific memory: the same token can carry different information at different depths, and the tensor does not compress well as one clean global object. That pushed us toward a local vector-quantization format instead of a global factorization.

We treat each `(token, layer)` PLE row as a 256-dimensional vector and split it into small 8-dimensional groups. For every `(layer, group)` pair, we learn a compact 128-entry vector-quantization codebook over these 8D blocks.
Each PLE row is then reconstructed as a sequence of independent codebook lookups. The codebooks themselves are tiny compared to the assignment payload, so most of the storage is spent on the compact per-row code assignments.

The assignment/refit step is sensitivity-weighted. For ordinary linear layers, local curvature can often be estimated from activations with the familiar `X^T X` approximation. PLE does not give the same clean dense activation view, because its rows are selected by discrete token/layer indices and used very unevenly.
We therefore estimate a small empirical Fisher-style sensitivity metric from gradients for each `(layer, group)`. Since raw gradient statistics are noisy, especially around rare tokens, we use a regularized shared metric rather than per-token sensitivity. This keeps the format simple while making the VQ assignments prefer errors in directions the model is less sensitive to, instead of treating all coordinate-space errors equally.

### PLE storage-quality tradeoff

The goal of the PLE codec is not to beat scalar W3/W4 at equal KL; it is to make the PLE payload small enough to fit the edge budget while keeping the full checkpoint usable. The table below isolates PLE only.

| Method | PLE size | KLD | KL p95 | Top1 |
| --- | --- | --- | --- | --- |
| Ours: obj-AQLM/Fisher b32 | 0.26 GB | 0.1231 | 0.5624 | 0.9184 |
| Direct W2 gs64 | 0.71 GB | 0.0536 | 0.2216 | 0.9462 |
| Direct W3 gs64 | 1.02 GB | 0.0119 | 0.0496 | 0.9729 |
| Direct W4 gs64 | 1.33 GB | 0.0032 | 0.0157 | 0.9857 |

Direct scalar quantization gives lower PLE-only KL, but at a much larger PLE payload. The AQLM-style codec is the point that makes the final small-checkpoint budget possible.

#### Compressing Token Embeddings/LM head

Our token-embedding scheme follows the same spirit as ggml's k-quants in llama.cpp: we search the per-group scale and offset that minimize reconstruction error. For the MLX release path, we keep the representation flat rather than hierarchical: group sizes are 32–128, and each group carries its own scale and zero point directly. This keeps the token embedding / LM-head path aligned with the same runtime contract as the transformer weights, while GGUF and Unsloth checkpoints remain the right public baselines for llama.cpp-style deployment.

## Bit-width schedule

---

Not every layer deserves the same precision. The last stage of the pipeline decides, per transformer module, how many bits and what group size it gets, so the whole model lands under a target size while giving up as little quality as possible.

We first build a small bank of full checkpoints at different settings — `w2_gs32`, `w3_gs128`, `w3_gs64`, `w3_gs32`, `w4_gs128`, `w4_gs64` — all produced by the same GPTQ/QEP/mseclip recipe. A scheduler then assigns each module one option from the bank under a global byte budget.

The obvious approach is to score each module's sensitivity on its own and fill the budget greedily. The catch is that the real objective — how much the *whole* model degrades — doesn't split cleanly into independent per-module costs, especially once the average drops below 4 bits, where errors in different modules start interacting. A per-module proxy quietly stops tracking the thing you care about.

So we use RCO (Riemannian Constrained Optimization) [arXiv:2605.00649], which optimizes the real loss directly. The discrete "which option" choice is relaxed into a soft mixture over the bank, and the size budget is treated as a constraint the optimizer is never allowed to leave — every gradient step is projected back onto the exact byte target, so the search only ever explores configurations that would actually ship. We tune the allocation against the true KL-to-dense signal under an exact size budget, with no penalty weights to babysit, and re-running for a different target size is cheap.

One detail worth stating plainly: RCO searches over a bank of already-quantized anchors to learn the schedule, but we don't ship that splice. The release checkpoint is re-quantized from the dense model in a single GPTQ/QEP pass using the learned schedule, so every module comes from one consistent run rather than a patchwork of independently quantized pieces.

### RCO ablation

To show that schedule search matters, we compare a uniform checkpoint, the RCO-selected assignment at the same size target, and the final scheduled requant from the dense model.

|  | KL mean | KL p95 |
| --- | --- | --- |
| Base Uniform | 0.4313 | 2.1904 |
| RCO | 0.3147 | 1.5854 |
| Requantization | 0.3147 | 1.5427 |

The scheduled requant keeps the RCO quality gain while avoiding a release artifact assembled as a splice of independently quantized bank checkpoints.

## Implementation in qlip

---

All compression runs were implemented in `qlip`, thestage's internal neural-network compression framework. `qlip` provides a unified interface
for model loading, calibration data, module selection, compression passes, artifact materialization, and evaluation, which makes it easy to add
new compression algorithms without rebuilding the full pipeline around each method.

For Gemma 4, this allowed us to compose several primitives in one reproducible flow: GPTQ/QEP for transformer projections, an AQLM-style codec
for PLE, scalar quantization for token embeddings, and Riemannian Constrained Optimization for non-uniform bit-width and group-size schedules.
The same framework is used to materialize the final release checkpoint from the dense model with the learned schedule, rather than treating the
schedule as a one-off experimental artifact.

# Experiments

---

![Model artifact size vs KL](https://cdn.thestage.ai/production/cms_file_upload/1780410507-019c0eac-f163-4c06-a48a-95abc3a0da75/gemma4-pareto-plain-kl-size-2026-06-01%20(1).svg)

Figure 1 compares final release artifacts against public GGUF checkpoints. The x-axis is final artifact size, not dense parameter count; the y-axis is held-out teacher KL, so lower is better. Red points are our scheduled-requant release checkpoints, not intermediate RCO bank splices.

## Experiments setup

We keep three kinds of data separate: calibration data that builds the compressed checkpoints, a distribution-proxy benchmark for fast internal tracking, and public benchmarks for the headline numbers.

Calibration is not a generic Wikitext setup but an explicit Gemma 4 mix — self-generated model traces, synthetic instruction and domain prompts, and multi-turn conversations, with a thin slice of public data for coverage. The rows are frozen as fixed views, so every GPTQ/QEP and scheduled-requant pass sees the same examples.

The distribution-proxy benchmark asks how far a compressed model drifts from the BF16 teacher rather than chasing a leaderboard. Its prompts are generated — never copied from the public test sets — and balanced across eight categories: instruction following, multiple-choice reasoning, math and science, code, tool use, general assistant queries, multilingual, and long multi-turn. For each prompt the teacher generates a reference path, and we measure the KL between the teacher's and the compressed model's next-token distributions along it, tracking mean and tail KL and top-1 agreement. This same teacher KL is the objective RCO optimizes when it searches the bit schedule: the differentiable pass runs 1000 steps at lr 0.02 on 512 chat-template train views (128 held back for validation) at a 512-token context, annealing its Gumbel sampling temperature from 0.7 down to 0.05. A disjoint set of 512 held-out proxy prompts then checks that the chosen schedule transfers rather than overfitting the train views.

For the public numbers we equalize the backend: every model — ours and the GGUF baselines alike — is dequantized to a standard BF16 checkpoint and served through vLLM, instead of mixing MLX, llama.cpp, custom kernels, and HF in a single table. We report MMLU-Pro for general knowledge, IFEval for instruction following, and Tau2 for multi-step tool use. For Tau2 the Gemma checkpoint under test acts as the agent, while the simulated user is Qwen3-235B-A22B-2507, so the task environment stays fixed and only the agent changes.

## Results

We evaluate final release checkpoints, not bank anchors or materialized RCO search artifacts. Public baselines are Unsloth GGUF checkpoints evaluated through the same BF16/vLLM task-eval path after dequantization. These tables report task quality; the Pareto figure above reports artifact size versus held-out teacher KL.

`Ours L` and `Ours M` are two release operating points: `L` keeps more quality at a larger artifact size, while `M` is the smaller release target used for the headline compression point. Both are final scheduled-requant checkpoints produced from the dense model, not RCO bank materializations.

### E2B

| Model | Compression ratio | MMLU Pro | IFeval | Tau2* (avg over 3) |
| --- | --- | --- | --- | --- |
| BF16 | 1.00× | 61.85 | 74.68 | 30.67 |
| Ours L | 5.62**×** | **54.48** | **74.86** | 22.20 |
| Ours M | **6.40×** | 49.85 | 71.53 | **23.45** |
| Unsloth Q3-K-S | 3.81× | 48.20 | 64.51 | 18.69 |
| Unsloth UD-Q2-K-XL | 3.87× | 43.17 | 66.54 | 20.23 |

### E4B

| Model | Compression ratio | MMLU Pro | IFeval | Tau2* |
| --- | --- | --- | --- | --- |
| BF16 | 1.00× | 70.49 | 81.33 | 37.19 |
| Ours L | 4.64**×** | 67.41 | 81.52 | **33.25** |
| Ours M | **5.60×** | 63.54 | **80.78** | 29.04 |
| Unsloth Q3-K-S | 3.90× | **63.66** | 77.08 | 30.47 |
| Unsloth UD-Q2-K-XL | 4.01× | 58.69 | 79.67 | 22.91 |

*Computed with Qwen3-235B-A22B-2507 as user simulator

## Performance benchmarks

Runtime benchmarking is tracked separately from the quality tables above. The release claim in this post is about artifact size and task quality under an MLX-compatible compression format; TTFT, decode throughput, and peak-memory comparisons against Unsloth GGUF/llama.cpp and a uniform MLX W4 baseline should be reported here once the final device measurements are locked.

## References

### Models and runtime formats

- Google. **Gemma 4 E2B / E4B model cards.**
    
    https://huggingface.co/google/gemma-4-E2B-it
    
    https://huggingface.co/google/gemma-4-E4B-it
    
- MLX Contributors. **MLX quantization API documentation.**
    
    https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html
    
- Hugging Face. **GGUF format documentation.**
    
    https://huggingface.co/docs/transformers/main/en/gguf
    
- Unsloth. **Gemma 4 GGUF checkpoints used as public baselines.**
    
    https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF
    
    https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF
    

### Compression methods

- Frantar, E., Ashkboos, S., Hoefler, T., Alistarh, D. **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.** ICLR 2023.
    
    https://arxiv.org/abs/2210.17323
    
- Arai, Y., Ichikawa, Y. **Quantization Error Propagation: Revisiting Layer-Wise Post-Training Quantization.** arXiv:2504.09629, 2025.
    
    https://arxiv.org/abs/2504.09629
    
- Egiazarian, V., Panferov, A., Kuznedelev, D., Frantar, E., Babenko, A., Alistarh, D. **Extreme Compression of Large Language Models via Additive Quantization.** ICML 2024.
    
    https://arxiv.org/abs/2401.06118
    
- Jégou, H., Douze, M., Schmid, C. **Product Quantization for Nearest Neighbor Search.** IEEE TPAMI, 2011.
    
    https://doi.org/10.1109/TPAMI.2010.57
    
- Helcig, M., Alistarh, D. **Model Compression with Exact Budget Constraints via Riemannian Manifolds.** arXiv:2605.00649, 2026.
    
    https://arxiv.org/abs/2605.00649
    

### Evaluation

- Wang, Y. et al. **MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark.** NeurIPS 2024.
    
    https://arxiv.org/abs/2406.01574
    
- Zhou, J. et al. **Instruction-Following Evaluation for Large Language Models.** arXiv:2311.07911, 2023.
    
    https://arxiv.org/abs/2311.07911
    
- Barres, V., Dong, H., Ray, S., Si, X., Narasimhan, K. **τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment.** arXiv:2506.07982, 2025.
    
    https://arxiv.org/abs/2506.07982