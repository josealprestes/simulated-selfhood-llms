import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

# Carregar os dados
with open("analysis_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)
df["logical_consistency"] = 1 - df["contradiction_rate"]

grouped = df.groupby(["model", "category"]).agg({
    "semantic_similarity": "mean",
    "textual_similarity": "mean",
    "logical_consistency": "mean"
}).reset_index()

categories = sorted(df["category"].unique())
metrics = ["semantic_similarity", "textual_similarity", "logical_consistency"]
models = grouped["model"].unique()
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]

fig, axs = plt.subplots(2, 3, subplot_kw=dict(polar=True), figsize=(18, 10))
axs = axs.flatten()

for idx, model in enumerate(models):
    ax = axs[idx]
    model_data = grouped[grouped["model"] == model]

    for cat in categories:
        values = model_data[model_data["category"] == cat][metrics].values
        if values.size == 0:
            continue
        stats = values.flatten().tolist()
        stats += stats[:1]
        ax.plot(angles, stats, label=cat)
        ax.fill(angles, stats, alpha=0.1)

    ax.set_title(f"{model} – Consistency by Category")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(["Semantic", "Textual", "Logical"])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"])

axs[-1].legend(loc='center', bbox_to_anchor=(0.5, -0.2), ncol=3)
fig.suptitle("Self-Reference Consistency per Category (Radar per Model)", fontsize=16)
fig.tight_layout()

plt.savefig("Figure_2.pdf", format='pdf', bbox_inches='tight')
plt.show()
