"""Local MLX quality diagnostics using lm-evaluation-harness.

This script evaluates the MLX runtime directly and is useful for smoke checks
and local debugging. Release headline quality numbers use the production
standard-HF-checkpoint + vLLM protocols under benchmarks/quality instead.

Usage:
    python benchmarks/evaluate.py --tasks ifeval --limit 10
    python benchmarks/evaluate.py --tasks ifeval --apply-chat-template --max-tokens 2048
    python benchmarks/evaluate.py --tasks ifeval gsm8k --output-dir ./eval_results
"""

import argparse
import collections
import copy
import json
import logging
import os
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Optional

import lm_eval
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from lm_eval.api.model import LM
from tqdm import tqdm

from edge_lm.models.load import load, DEFAULT_MODEL

DEFAULT_MAX_TOKENS = 8192


def _rstrip_until(s, untils):
    l = len(s)
    f = [s.find(u) for u in untils]
    f = [l if x < 0 else x for x in f]
    return s[: min(f)]


def _pad_inputs(inputs):
    lengths = np.array([len(x) for x in inputs])
    maxlen = lengths.max()
    padded = np.stack(
        [np.pad(x, (0, maxlen - len(x))) for x in inputs],
        axis=0,
    )
    return mx.array(padded), mx.array(lengths)


class TheStageMLX(LM):
    """lm-eval backend for our custom MLX model with compact PLE."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        max_tokens: Optional[int] = None,
        batch_size: int = 8,
        use_chat_template: Optional[bool] = None,
    ):
        super().__init__()
        self._model, self.tokenizer = load(model_id)
        self._max_tokens = max_tokens
        self._batch_size = batch_size
        self.use_chat_template = use_chat_template
        if use_chat_template is None:
            self.use_chat_template = getattr(self.tokenizer, "chat_template", None) is not None

    def apply_chat_template(self, chat_history, add_generation_prompt=True):
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )

    def _forward(self, input_ids, cache=None):
        """Forward pass returning logits."""
        logits = self._model(input_ids, cache=cache)
        if hasattr(logits, "logits"):
            logits = logits.logits
        return logits

    def _process_prompt(self, prompt, step_size=2048):
        prompt = mx.array(prompt)[None]
        cache = self._model.language_model.make_cache()
        for i in range(0, prompt.shape[1], step_size):
            logits = self._forward(prompt[:, i : i + step_size], cache=cache)
            mx.eval(logits)
            mx.clear_cache()
        logprobs = nn.log_softmax(logits[:, -1, :].astype(mx.float32))
        return logprobs, cache

    def _score_fn(self, inputs, cache=None, step_size=2048):
        inputs, lengths = _pad_inputs(inputs)
        inputs, targets = inputs[..., :-1], inputs[..., 1:]

        cache = cache or self._model.language_model.make_cache()
        offset = 0
        scores, is_greedy = [], []
        for i in range(0, inputs.shape[1], step_size):
            inp = inputs[:, i : i + step_size]
            T = inp.shape[1]

            logits = self._forward(inp, cache=cache)
            log_probs = nn.log_softmax(logits.astype(mx.float32))

            score = mx.take_along_axis(
                log_probs, targets[:, i : i + step_size, mx.newaxis], axis=-1
            )[..., 0]

            ig = targets[:, i : i + step_size] == mx.argmax(logits, axis=-1)
            ig = mx.where(mx.arange(offset, T + offset) < lengths[:, None], ig, False)

            mx.eval(score, ig)
            mx.clear_cache()

            is_greedy.append(ig)
            scores.append(score)
            offset += T

        scores = mx.concatenate(scores, axis=1)
        is_greedy = mx.concatenate(is_greedy, axis=1)
        return scores, lengths, is_greedy

    def _tokenize(self, texts):
        return [
            tuple(self.tokenizer.encode(t, add_special_tokens=not self.use_chat_template))
            for t in texts
        ]

    @property
    def tokenizer_name(self):
        return self.tokenizer.name_or_path.replace("/", "__")

    def loglikelihood(self, requests):
        logging.info(f"Estimating loglikelihood for {len(requests)} pairs.")

        group_reqs = collections.defaultdict(list)
        for idx, req in enumerate(requests):
            group_reqs[req.args[0]].append((idx, req.args[1]))

        questions = list(group_reqs.keys())
        responses = [list(zip(*group_reqs[q])) for q in questions]
        indices = [r[0] for r in responses]
        responses = [r[1] for r in responses]

        scores, is_greedy = [], []
        for q, rs in tqdm(zip(questions, responses), total=len(questions)):
            prefix = self._tokenize([q])[0]
            full_sequences = self._tokenize([q + r for r in rs])

            max_tokens = self._max_tokens or DEFAULT_MAX_TOKENS
            max_completed_l = max(len(s) for s in full_sequences)
            truncation = max(0, max_completed_l - max_tokens - 1)
            prefix_l = max(len(prefix) - truncation, 0)
            prefix = prefix[len(prefix) - prefix_l :]

            if prefix_l == 0:
                scores.extend([-float("inf")] * len(rs))
                is_greedy.extend([False] * len(rs))
                continue

            logprobs, cache = self._process_prompt(prefix)
            max_idx = mx.argmax(logprobs).item()

            for s in full_sequences:
                inputs = s[len(prefix) :]
                scores.append(logprobs[0, inputs[0]].item())
                is_greedy.append(inputs[0] == max_idx)

                if len(inputs) == 1:
                    continue
                score, _, ig = self._score_fn(
                    mx.array(inputs)[None, :], cache=copy.deepcopy(cache)
                )
                scores[-1] += mx.sum(score).item()
                is_greedy[-1] &= mx.all(ig).item()

        # Re-order to match original request order
        result = [None] * len(requests)
        score_idx = 0
        for idx_group in indices:
            for idx in idx_group:
                result[idx] = (scores[score_idx], is_greedy[score_idx])
                score_idx += 1

        return result

    def loglikelihood_rolling(self, requests):
        logging.info(f"Estimating loglikelihood rolling for {len(requests)} sequences.")
        inputs = self._tokenize([req.args[0] for req in requests])
        all_scores = []
        for i in tqdm(range(0, len(inputs), self._batch_size)):
            batch = inputs[i : i + self._batch_size]
            scores, lengths, _ = self._score_fn(batch)
            mask = mx.arange(scores.shape[-1]) < lengths[:, None]
            all_scores.extend((mask * scores).sum(axis=-1).tolist())
        return all_scores

    def generate_until(self, requests):
        logging.info(f"Generating continuation for {len(requests)} sequences.")
        contexts, options = zip(*[req.args for req in requests])

        completions = []
        for context, opt in tqdm(zip(contexts, options), total=len(contexts)):
            tokens = self.tokenizer.encode(
                context, add_special_tokens=not self.use_chat_template
            )
            max_tokens = self._max_tokens or opt.get("max_gen_tokens", DEFAULT_MAX_TOKENS)

            cache = self._model.language_model.make_cache()

            # Process prompt in chunks
            input_ids = mx.array(tokens)[None]
            for i in range(0, input_ids.shape[1], 2048):
                logits = self._forward(input_ids[:, i : i + 2048], cache=cache)
                mx.eval(logits)
                mx.clear_cache()

            # Generate
            generated = []
            for _ in range(max_tokens):
                next_token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
                if next_token == self.tokenizer.eos_token_id:
                    break
                generated.append(next_token)
                logits = self._forward(mx.array([[next_token]]), cache=cache)
                mx.eval(logits)

            text = self.tokenizer.decode(generated)
            untils = opt.get("until", [])
            if untils:
                text = _rstrip_until(text, untils)
            completions.append(text)

        return completions


def main():
    parser = argparse.ArgumentParser(description="Evaluate model quality with lm-eval")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--output-dir", default="eval_results")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=None, help="Limit examples per task")
    parser.add_argument("--num-shots", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--apply-chat-template",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--fewshot-as-multiturn", action="store_true", default=False)
    parser.add_argument("--use-ref", action="store_true", help="Use original HF model via mlx-lm MLXLM backend")
    parser.add_argument("--hf-model", type=str, default="google/gemma-4-E2B-it")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    mx.random.seed(args.seed)

    if args.use_ref:
        print(f"Loading reference model {args.hf_model} via mlx-vlm...")
        from mlx_vlm import load as load_vlm
        ref_model, ref_processor = load_vlm(args.hf_model)
        lm = TheStageMLX.__new__(TheStageMLX)
        LM.__init__(lm)
        lm._model = ref_model
        lm.tokenizer = ref_processor.tokenizer
        lm._max_tokens = args.max_tokens
        lm._batch_size = args.batch_size
        lm.use_chat_template = args.apply_chat_template
        if lm.use_chat_template is None:
            lm.use_chat_template = getattr(lm.tokenizer, "chat_template", None) is not None
    else:
        print(f"Loading model {args.model}...")
        lm = TheStageMLX(
            model_id=args.model,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
            use_chat_template=args.apply_chat_template,
        )

    print(f"Running evaluation: {args.tasks}")
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=args.tasks,
        apply_chat_template=lm.use_chat_template,
        num_fewshot=args.num_shots,
        limit=args.limit,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        fewshot_random_seed=args.seed,
    )

    # Save results
    filename = f"eval_{'_'.join(args.tasks)}.json"
    output_path = output_dir / filename
    output_path.write_text(json.dumps(results["results"], indent=2))

    print(f"\nResults saved to {output_path}")
    for task, result in results["results"].items():
        print(f"\n{task}:")
        for k, v in result.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            elif k != "alias":
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
