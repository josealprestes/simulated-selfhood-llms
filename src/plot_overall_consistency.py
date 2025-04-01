import pandas as pd
import matplotlib.pyplot as plt

# Carrega os dados
df = pd.read_csv("analysis_results.csv")

# Calcula a média de cada métrica por modelo
mean_metrics = df.groupby("model")[["semantic_similarity", "textual_similarity", "contradiction_rate"]].mean().reset_index()
mean_metrics = mean_metrics.rename(columns={
    "semantic_similarity": "Semantic Similarity",
    "textual_similarity": "Textual Similarity",
    "contradiction_rate": "Logical Consistency"
})
# Converte taxa de contradição em consistência lógica
mean_metrics["Logical Consistency"] = 1 - mean_metrics["Logical Consistency"]

# Gráfico de barras
mean_metrics.set_index("model").plot(kind="bar", figsize=(10, 6))
plt.title("Average Self-Reference Consistency per Model")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("Figure_3.pdf", format='pdf', bbox_inches='tight')
plt.show()
