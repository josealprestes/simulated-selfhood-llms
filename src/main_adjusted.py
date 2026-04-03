import os
import json
import csv
import time
import gc
import argparse
from datetime import datetime
from pathlib import Path

from llama_cpp import Llama

# ========== DEFAULT CONFIGURATION ==========
MODELS = {
    "hermes": "Hermes-3-Llama-3.2-3B.Q4_K_M.gguf",
    "mistral": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    "tinyllama": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    "openchat": "openchat-3.5-0106.Q4_K_M.gguf",
    "stablelm": "stablelm-zephyr-3b.Q4_K_M.gguf",
}

DEFAULT_REPETITIONS = 10
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
DEFAULT_MAX_TOKENS = 100
DEFAULT_CTX_SIZE = 2048
DEFAULT_THREADS = 4
DEFAULT_SLEEP = 0.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the self-reference experiment on one or more local GGUF models."
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Run a specific model only (e.g. --model mistral). Omit to run all models.",
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--ctx_size", type=int, default=DEFAULT_CTX_SIZE)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument(
        "--models_dir",
        type=str,
        default="models",
        help=(
            "Preferred directory containing the GGUF model files. "
            "If a model is not found there, the script will also try the current directory."
        ),
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default="prompts.json",
        help="Path to the prompts.json file.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs",
        help="Root directory under which run folders will be created.",
    )
    parser.add_argument(
        "--run_tag",
        type=str,
        default=None,
        help=(
            "Optional run folder name, e.g. temp_0_2. "
            "If omitted, one will be generated from temperature/top_p."
        ),
    )
    parser.add_argument(
        "--sleep_seconds",
        type=float,
        default=DEFAULT_SLEEP,
        help="Pause between generations to reduce local contention.",
    )
    return parser.parse_args()


def load_prompts(prompts_file: str) -> list[dict]:
    with open(prompts_file, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("prompts.json must contain a non-empty list of prompt records.")
    return prompts


def build_output_dir(output_root: str, run_tag: str | None, temperature: float, top_p: float) -> Path:
    if run_tag is None:
        run_tag = f"temp_{str(temperature).replace('.', '_')}_top_p_{str(top_p).replace('.', '_')}"
    output_dir = Path(output_root) / run_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def resolve_model_path(models_dir: str, model_file: str) -> Path:
    """
    Try to find the model file first in models_dir, then in the current directory.
    """
    candidates = [
        Path(models_dir) / model_file,
        Path(model_file),
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    searched = "\n".join(str(c.resolve()) for c in candidates)
    raise FileNotFoundError(
        f"Model file not found: {model_file}\nSearched in:\n{searched}"
    )


def query_model(model: Llama, prompt: str, max_tokens: int, temperature: float, top_p: float) -> str:
    output = model(
        f"Question: {prompt}\nAnswer:",
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return output["choices"][0]["text"].strip()


def run_experiment(
    model_name: str,
    model_file: str,
    prompts: list[dict],
    models_dir: str,
    output_dir: Path,
    temperature: float,
    top_p: float,
    max_tokens: int,
    repetitions: int,
    ctx_size: int,
    threads: int,
    sleep_seconds: float,
) -> None:
    model_path = resolve_model_path(models_dir, model_file)

    print(
        f"\nRunning: {model_name} | temp={temperature} | top_p={top_p} | "
        f"max_tokens={max_tokens} | repetitions={repetitions}"
    )
    print(f"Model path: {model_path}")

    model = Llama(model_path=str(model_path), n_ctx=ctx_size, n_threads=threads)

    results: list[dict] = []
    for entry in prompts:
        prompt = entry["prompt"]
        category = entry["category"]
        for i in range(repetitions):
            response = query_model(model, prompt, max_tokens, temperature, top_p)
            results.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "model": model_name,
                    "category": category,
                    "prompt": prompt,
                    "repetition": i + 1,
                    "response": response,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                }
            )
            print(f"[{model_name}] {prompt} -> {response[:80]}...")
            time.sleep(sleep_seconds)

    suffix = f"temp_{str(temperature).replace('.', '_')}_top_p_{str(top_p).replace('.', '_')}"
    json_path = output_dir / f"self_reference_{model_name}_{suffix}.json"
    csv_path = output_dir / f"self_reference_{model_name}_{suffix}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"Finished: {model_name}. Results saved to {output_dir}")
    del model
    gc.collect()


def main() -> None:
    args = parse_args()
    prompts = load_prompts(args.prompts_file)
    output_dir = build_output_dir(args.output_root, args.run_tag, args.temperature, args.top_p)

    if args.model:
        model_key = args.model.lower()
        if model_key not in MODELS:
            raise ValueError(f"Unknown model '{model_key}'. Available: {list(MODELS.keys())}")
        selected_models = {model_key: MODELS[model_key]}
    else:
        selected_models = MODELS

    for idx, (model_key, model_file) in enumerate(selected_models.items(), start=1):
        run_experiment(
            model_name=model_key,
            model_file=model_file,
            prompts=prompts,
            models_dir=args.models_dir,
            output_dir=output_dir,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            repetitions=args.repetitions,
            ctx_size=args.ctx_size,
            threads=args.threads,
            sleep_seconds=args.sleep_seconds,
        )
        if idx < len(selected_models):
            print(f"Memory cleared after {model_key}.\n")
            time.sleep(3)

    print("All experiments completed successfully.")


if __name__ == "__main__":
    main()