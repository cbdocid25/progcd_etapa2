import pandas as pd
from sklearn.decomposition import PCA

# 1. Carregar o dataset normalizado
df = pd.read_csv("../../model/dataset/creditcard.csv")

# 2. Selecionar as colunas V1 a V28
colunas_v = [f"V{i}" for i in range(1, 29)]
X = df[colunas_v]

# 3. Aplicar PCA com 10 componentes
pca = PCA(n_components=20)
pca_result = pca.fit_transform(X)

# 4. Criar novo DataFrame com os 10 componentes PCA
colunas_pca = [f"PCA{i+1}" for i in range(20)]
df_pca = pd.DataFrame(pca_result, columns=colunas_pca)

# 5. Anexar Time, Amount e Class ao novo DataFrame
df_pca['Time'] = df['Time'].values
df_pca['Amount'] = df['Amount'].values
df_pca['Class'] = df['Class'].values

# 6. Salvar o novo dataset com PCA aplicado
df_pca.to_csv("dataset/dataset_com_pca.csv", index=False)

# 7. Variância explicada
variancia_2 = pca.explained_variance_ratio_[:2].sum()
variancia_total = pca.explained_variance_ratio_.sum()

print(f"✅ Variância explicada por PCA1 e PCA2: {variancia_2:.2%}")
print(f"✅ Variância explicada pelos 10 componentes: {variancia_total:.2%}")

print("> 80%	     Os 2 componentes retêm quase tudo → ótima redução")
print("= 50% ~ 80%  Redução razoável, mas pode perder nuances importantes")
print("< 30%	🚨   Pouca informação preservada nos 2 primeiros eixos")
print()
print("✅ Dataset com os 10 componentes principais salvo como 'dataset_com_pca.csv'")
print()
print(df_pca.head())
