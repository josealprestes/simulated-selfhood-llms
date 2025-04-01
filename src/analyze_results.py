# analyze_results.py

import os, glob, csv, json
import pandas as pd
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# ========== CONFIGURATION ==========

RESULTS_DIR = "outputs"
TEXTUAL_THRESHOLD = 0.85
SEMANTIC_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
NLI_MODEL_NAME = "roberta-large-mnli"

# ========== LOAD MODELS ==========

print("🔍 Loading Sentence-BERT for semantic similarity...")
sbert = SentenceTransformer(SEMANTIC_MODEL_NAME)

print("🔍 Loading RoBERTa for natural language inference (NLI)...")
nli = pipeline("text-classification", model=NLI_MODEL_NAME)

# ========== TEXTUAL SIMILARITY ==========

def compute_textual_similarity(responses):
    base = responses[0]
    scores = [SequenceMatcher(None, base, r).ratio() for r in responses[1:]]
    return sum(scores) / len(scores) if scores else 0

# ========== SEMANTIC SIMILARITY ==========

def compute_semantic_similarity(responses):
    embeddings = sbert.encode(responses, convert_to_tensor=True)
    base = embeddings[0]
    scores = [util.cos_sim(base, e).item() for e in embeddings[1:]]
    return sum(scores) / len(scores) if scores else 0

# ========== CONTRADICTION RATE ==========

def compute_contradiction_rate(responses):
    contradictions = 0
    comparisons = 0
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            result = nli(f"{responses[i]} </s> {responses[j]}")[0]
            if result["label"] == "CONTRADICTION":
                contradictions += 1
            comparisons += 1
    return contradictions / comparisons if comparisons > 0 else 0

# ========== ANALYZE A SINGLE MODEL FILE ==========

def analyze_model_file(file_path):
    df = pd.read_csv(file_path, encoding="cp1252")
    grouped = df.groupby("prompt")
    results = []

    for prompt, group in grouped:
        responses = group["response"].tolist()
        model = group["model"].iloc[0]
        category = group["category"].iloc[0]

        textual = compute_textual_similarity(responses)
        semantic = compute_semantic_similarity(responses)
        contradiction = compute_contradiction_rate(responses)

        results.append({
            "model": model,
            "category": category,
            "prompt": prompt,
            "textual_similarity": round(textual, 4),
            "semantic_similarity": round(semantic, 4),
            "contradiction_rate": round(contradiction, 4)
        })

    return results

# ========== MAIN LOOP ==========

all_results = []

for file in glob.glob(os.path.join(RESULTS_DIR, "*.csv")):
    print(f"\n📂 Analyzing file: {file}")
    results = analyze_model_file(file)
    all_results.extend(results)

# ========== EXPORT RESULTS ==========

df_results = pd.DataFrame(all_results)
df_results.to_csv("analysis_results.csv", index=False)
with open("analysis_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("\n✅ Analysis completed!")
print("📁 Saved: analysis_results.csv and analysis_results.json")
