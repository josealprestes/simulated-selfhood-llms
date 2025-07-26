# run_llm_experiment.py

import os, time, json, csv, gc
from datetime import datetime
from llama_cpp import Llama

# ==================== CONFIGURATION ====================
MODELS = {
    "hermes":    "Hermes-3-Llama-3.2-3B.Q4_K_M.gguf",
    "mistral":   "mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    "tinyllama": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    "openchat":  "openchat-3.5-0106.Q4_K_M.gguf",
    "stablelm":  "stablelm-zephyr-3b.Q4_K_M.gguf"
}

REPETITIONS = 10
TEMPERATURE = 0.7
TOP_P = 0.95
MAX_TOKENS = 100
CTX_SIZE = 2048

# ==================== LOAD PROMPTS ====================
with open("prompts.json", "r") as f:
    prompts = json.load(f)

# ==================== MODEL QUERY FUNCTION ====================
def query_model(model, prompt):
    output = model(f"Question: {prompt}\nAnswer:",
                   max_tokens=MAX_TOKENS,
                   temperature=TEMPERATURE,
                   top_p=TOP_P)
    return output["choices"][0]["text"].strip()

# ==================== EXPERIMENT LOOP ====================
def run_experiment(model_name, model_path):
    print(f"\n🚀 Running model: {model_name}")
    model = Llama(model_path=os.path.join("models", model_path), n_ctx=CTX_SIZE, n_threads=4)

    results = []
    for entry in prompts:
        for i in range(REPETITIONS):
            response = query_model(model, entry["prompt"])
            results.append({
                "timestamp": datetime.utcnow().isoformat(),
                "model": model_name,
                "category": entry["category"],
                "prompt": entry["prompt"],
                "repetition": i + 1,
                "response": response,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "max_tokens": MAX_TOKENS
            })
            print(f"[{model_name}] {entry['prompt']} → {response[:60]}...")
            time.sleep(0.2)

    # Ensure output directory exists
    os.makedirs("outputs", exist_ok=True)

    # Save results
    json_path = f"outputs/self_reference_{model_name}.json"
    csv_path  = f"outputs/self_reference_{model_name}.csv"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    with open(csv_path, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"✅ Completed: {model_name}. Results saved to 'outputs/'")
    del model
    gc.collect()

# ==================== RUN ALL MODELS ====================
for model_key, file_name in MODELS.items():
    run_experiment(model_key, file_name)
    print(f"🧹 Memory cleared after {model_key}...\n")
    time.sleep(3)

print("🏁 All models completed successfully!")
