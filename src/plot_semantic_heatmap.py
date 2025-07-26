import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar os dados do arquivo CSV
df = pd.read_csv("analysis_results.csv")

# === Heatmap de CONSISTÊNCIA SEMÂNTICA ===
# Alteração 1: Usar 'semantic_similarity' como valor
heatmap_data = df.pivot_table(index="model", columns="category", values="semantic_similarity", aggfunc="mean")

# Alteração 2: A linha de conversão "1 - heatmap_data" foi REMOVIDA

plt.figure(figsize=(10, 6))
# Alteração 3 e 4: Título do gráfico e da barra de cores ajustados
sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", vmin=0, vmax=1, cbar_kws={"label": "Semantic Consistency"})
plt.title("Semantic Consistency by Model and Category")
plt.ylabel("Model")
plt.xlabel("Category")
plt.tight_layout()
# Alteração 5: Nome do arquivo salvo
plt.savefig("semantic_heatmap.pdf", format='pdf', bbox_inches='tight')
plt.show()