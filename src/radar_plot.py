import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Carrega os dados
df = pd.read_csv("analysis_results.csv", encoding="utf-8")

# Agrupa por modelo e calcula a média das métricas
grouped = df.groupby("model").agg({
    "semantic_similarity": "mean",
    "textual_similarity": "mean",
    "contradiction_rate": "mean"
}).reset_index()

# Adiciona coerência lógica como 1 - taxa de contradição
grouped["logical_consistency"] = 1 - grouped["contradiction_rate"]

# Seleciona apenas as colunas desejadas para o radar
metrics = ["semantic_similarity", "textual_similarity", "logical_consistency"]
labels = ["Semantic Similarity", "Textual Similarity", "Logical Consistency"]

# Constrói o radar
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]  # Fecha o círculo

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for i, row in grouped.iterrows():
    values = row[metrics].tolist()
    values += values[:1]  # Fecha o gráfico
    ax.plot(angles, values, label=row["model"], linewidth=2)
    ax.fill(angles, values, alpha=0.1)

# Estética
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), labels)
ax.set_title("Self-Reference Consistency Radar (per Model)", size=14, pad=20)
ax.set_ylim(0, 1)
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.05))
ax.grid(True)

plt.tight_layout()
plt.savefig("Figure_1.pdf", format='pdf', bbox_inches='tight')
plt.show()
