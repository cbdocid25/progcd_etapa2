import pandas as pd
from sklearn.decomposition import PCA

# 1. Carregar o dataset normalizado
df = pd.read_csv("../dataset/creditcard.csv")

# 2. Selecionar dinamicamente as colunas V1 a Vn
colunas_v = [col for col in df.columns if col.startswith("V")]
X = df[colunas_v]

# 3. Aplicar PCA com número total de componentes
pca = PCA()
pca.fit(X)

# 4. Calcular variância acumulada
variancia_acumulada = pca.explained_variance_ratio_.cumsum()

# 5. Descobrir o número de componentes necessários para 80%
n_componentes_80 = next(i for i, total in enumerate(variancia_acumulada) if total >= 0.80) + 1

# 6. Mostrar resultado
print("📊 Variância explicada acumulada:")
for i, v in enumerate(variancia_acumulada, start=1):
    print(f"PCA{i}: {v:.2%}")
print(f"\n✅ Você precisa de {n_componentes_80} componentes para atingir 80% da variância explicada.")
