import os
import glob
import json
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

SEMANTIC_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
NLI_MODEL_NAME = "roberta-large-mnli"


def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return [make_json_serializable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if pd.isna(obj):
        return None
    return obj

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze one experimental condition of the self-reference study."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing the CSV outputs for a single run condition.",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        required=True,
        help="Prefix for exported analysis files, e.g. analysis_results_temp_0_2.",
    )
    parser.add_argument(
        "--glob_pattern",
        type=str,
        default="*.csv",
        help="Optional glob pattern for input files inside input_dir.",
    )
    return parser.parse_args()


def read_csv_robust(file_path: str) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "cp1252"]
    last_error = None
    for enc in encodings:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error  # type: ignore[misc]


print("Loading Sentence-BERT for semantic similarity...")
sbert = SentenceTransformer(SEMANTIC_MODEL_NAME)

print("Loading RoBERTa for natural language inference (NLI)...")
nli = pipeline("text-classification", model=NLI_MODEL_NAME)


def compute_textual_similarity_all_pairs(responses: list[str]) -> float:
    pairs = list(combinations(responses, 2))
    if not pairs:
        return 0.0
    scores = [SequenceMatcher(None, a, b).ratio() for a, b in pairs]
    return float(sum(scores) / len(scores))



def compute_semantic_similarity_all_pairs(responses: list[str]) -> float:
    pairs = list(combinations(range(len(responses)), 2))
    if not pairs:
        return 0.0
    embeddings = sbert.encode(responses, convert_to_tensor=True)
    scores = [util.cos_sim(embeddings[i], embeddings[j]).item() for i, j in pairs]
    return float(sum(scores) / len(scores))



def compute_contradiction_rate_all_pairs(responses: list[str]) -> float:
    contradictions = 0
    comparisons = 0
    for a, b in combinations(responses, 2):
        result = nli(f"{a} </s> {b}")[0]
        if result["label"] == "CONTRADICTION":
            contradictions += 1
        comparisons += 1
    return (contradictions / comparisons) if comparisons > 0 else 0.0



def compute_diachronic_textual_similarity(responses: list[str]) -> float:
    if len(responses) <= 1:
        return 0.0
    base = responses[0]
    scores = [SequenceMatcher(None, base, r).ratio() for r in responses[1:]]
    return float(sum(scores) / len(scores))



def compute_diachronic_semantic_similarity(responses: list[str]) -> float:
    if len(responses) <= 1:
        return 0.0
    embeddings = sbert.encode(responses, convert_to_tensor=True)
    base = embeddings[0]
    scores = [util.cos_sim(base, emb).item() for emb in embeddings[1:]]
    return float(sum(scores) / len(scores))



def analyze_model_file(file_path: str) -> list[dict]:
    df = read_csv_robust(file_path)
    required_cols = {"model", "category", "prompt", "response"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {file_path}: {sorted(missing)}")

    grouped = df.groupby("prompt", sort=False)
    results: list[dict] = []

    for prompt, group in grouped:
        responses = group["response"].astype(str).tolist()
        model = group["model"].iloc[0]
        category = group["category"].iloc[0]
        temperature = group["temperature"].iloc[0] if "temperature" in group.columns else None
        top_p = group["top_p"].iloc[0] if "top_p" in group.columns else None
        max_tokens = group["max_tokens"].iloc[0] if "max_tokens" in group.columns else None

        textual = compute_textual_similarity_all_pairs(responses)
        semantic = compute_semantic_similarity_all_pairs(responses)
        contradiction = compute_contradiction_rate_all_pairs(responses)
        dia_textual = compute_diachronic_textual_similarity(responses)
        dia_semantic = compute_diachronic_semantic_similarity(responses)

        results.append(
            {
                "model": model,
                "category": category,
                "prompt": prompt,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "textual_similarity": round(textual, 4),
                "semantic_similarity": round(semantic, 4),
                "contradiction_rate": round(contradiction, 4),
                "logical_consistency": round(1 - contradiction, 4),
                "diachronic_textual_similarity": round(dia_textual, 4),
                "diachronic_semantic_similarity": round(dia_semantic, 4),
            }
        )

    return results



def build_model_summary(df_results: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "textual_similarity",
        "semantic_similarity",
        "contradiction_rate",
        "logical_consistency",
        "diachronic_textual_similarity",
        "diachronic_semantic_similarity",
    ]
    summary = (
        df_results.groupby(["model", "temperature", "top_p"], dropna=False)[numeric_cols]
        .mean()
        .reset_index()
        .sort_values(["model", "temperature", "top_p"])
    )
    return summary



def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    all_results: list[dict] = []
    files = sorted(glob.glob(str(input_dir / args.glob_pattern)))
    if not files:
        raise FileNotFoundError(f"No files found in {input_dir} matching {args.glob_pattern}")

    for file in files:
        print(f"Analyzing file: {file}")
        all_results.extend(analyze_model_file(file))

    df_results = pd.DataFrame(all_results)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    csv_path = output_prefix.with_suffix(".csv")
    json_path = output_prefix.with_suffix(".json")
    summary_csv_path = output_prefix.parent / f"{output_prefix.stem}_model_summary.csv"
    summary_json_path = output_prefix.parent / f"{output_prefix.stem}_model_summary.json"

    df_results.to_csv(csv_path, index=False, encoding="utf-8")

    json_ready_results = make_json_serializable(all_results)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_ready_results, f, indent=2, ensure_ascii=False)

    df_summary = build_model_summary(df_results)
    df_summary.to_csv(summary_csv_path, index=False, encoding="utf-8")

    json_ready_summary = make_json_serializable(df_summary.to_dict(orient="records"))
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(json_ready_summary, f, indent=2, ensure_ascii=False)

    print("Analysis completed.")
    print(f"Saved prompt-level results to: {csv_path} and {json_path}")
    print(f"Saved model-level summary to: {summary_csv_path} and {summary_json_path}")


if __name__ == "__main__":
    main()