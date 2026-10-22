"""Versioned general-capability protocol for the pinned lm-eval 0.4.8 runner."""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

GSM_PROMPT = (
    "Question: {{question}}\nSolve this problem step by step. After your explanation, "
    "write ####, a space, and your final numeric answer together on the same line. "
    "Do not include a currency symbol or unit on that line.\nAnswer:"
)


def render_chat(self, chat_history, add_generation_prompt=True):
    rendered = self.tokenizer.apply_chat_template(
        chat_history, tokenize=False, add_generation_prompt=add_generation_prompt,
        continue_final_message=not add_generation_prompt, enable_thinking=False,
    )
    if add_generation_prompt and not rendered.endswith("<think>\n\n</think>\n\n"):
        raise RuntimeError("General evaluation requires a Qwen3 template with explicit non-thinking support")
    directory = os.environ.get("GENERAL_PROTOCOL_OUTPUT_DIR")
    if directory and not getattr(self, "_general_prompt_saved", False):
        (Path(directory) / "prompt_preview.txt").write_text(rendered, encoding="utf-8")
        self._general_prompt_saved = True
    return rendered


def configure_tasks(tasks, max_tokens):
    for name, task in tasks.items():
        if isinstance(task, dict):
            configure_tasks(task, max_tokens)
        elif task.get_config("output_type") == "generate_until":
            task.set_config("generation_kwargs", {
                "max_gen_toks": max_tokens, "temperature": 0.0, "do_sample": False,
            }, update=True)
            if task.get_config("task") == "gsm8k":
                task.set_config("doc_to_text", GSM_PROMPT)


def summarize_samples(root):
    report = {}
    for path in sorted(Path(root).rglob("samples_*.jsonl")):
        # Harness filenames end in an ISO timestamp; task names may contain '_'.
        name = path.name[len("samples_"):].split("_20", 1)[0].removesuffix(".jsonl")
        counts, invalid, seen = Counter(), Counter(), set()
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            if "[invalid]" in row.get("filtered_resps", []):
                invalid[row.get("filter", "none")] += 1
            key = row["doc_id"]
            if key in seen:
                continue
            seen.add(key)
            text = row["resps"][0][0]
            counts.update(samples=1, sid=int("<|sid_begin|>" in text),
                          empty=int(not text.strip()),
                          unclosed_think=int("<think>" in text and "</think>" not in text))
        report[name] = {**counts, "invalid_by_filter": dict(invalid)}
    return report


def install():
    if not os.environ.get("GENERAL_PROTOCOL_OUTPUT_DIR"):
        return
    from lm_eval import evaluator
    from lm_eval.models.vllm_causallms import VLLM

    if getattr(VLLM, "_general_protocol_installed", False):
        return
    VLLM._general_protocol_installed = True
    max_tokens = int(os.environ.get("GENERAL_MAX_NEW_TOKENS", "2048"))
    output_dir = Path(os.environ["GENERAL_PROTOCOL_OUTPUT_DIR"])
    VLLM.apply_chat_template = render_chat
    original_tasks = evaluator.get_task_dict

    def get_tasks(*args, **kwargs):
        tasks = original_tasks(*args, **kwargs)
        configure_tasks(tasks, max_tokens)
        return tasks

    evaluator.get_task_dict = get_tasks

    original_generate = VLLM._model_generate

    def generate(self, requests=None, generate=False, max_tokens=None, **kwargs):
        if generate and max_tokens >= self.max_length:
            raise ValueError("Generation budget must be smaller than model context length")
        outputs = original_generate(self, requests=requests, generate=generate,
                                    max_tokens=max_tokens, **kwargs)
        if generate:
            with (output_dir / "generation_diagnostics.jsonl").open("a", encoding="utf-8") as handle:
                for prompt, output in zip(requests, outputs):
                    completion = output.outputs[0]
                    record = {
                        "prompt_token_sha256": hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
                        "response_sha256": hashlib.sha256(completion.text.encode()).hexdigest(),
                        "generated_tokens": len(completion.token_ids),
                        "max_tokens": max_tokens,
                        "finish_reason": completion.finish_reason,
                        "stop_reason": completion.stop_reason,
                    }
                    handle.write(json.dumps(record) + "\n")
        return outputs

    VLLM._model_generate = generate
    print(f"[general] non-thinking, 0-shot, greedy, max_new_tokens={max_tokens}", flush=True)


def main():
    from lm_eval.__main__ import cli_evaluate
    install()
    return int(cli_evaluate() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
