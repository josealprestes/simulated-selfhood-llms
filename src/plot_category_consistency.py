import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar os dados do arquivo CSV
df = pd.read_csv("analysis_results.csv")

# === Heatmap por categoria ===
heatmap_data = df.pivot_table(index="model", columns="category", values="contradiction_rate", aggfunc="mean")
heatmap_data = 1 - heatmap_data  # converte para consistência

plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", vmin=0, vmax=1, cbar_kws={"label": "Logical Consistency"})
plt.title("Logical Consistency by Model and Category")
plt.ylabel("Model")
plt.xlabel("Category")
plt.tight_layout()
plt.savefig("Figure_4.pdf", format='pdf', bbox_inches='tight')
plt.show()
