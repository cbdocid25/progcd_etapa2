import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. Carregar o dataset bruto
df_raw = pd.read_csv("dataset/dataset_simulado.csv")

# Calcula a matriz de correlação de Pearson entre essas duas colunas com
#  1.0	Forte correlação positiva
# -1.0	Forte correlação negativa
#  0	Sem correlação (independência estatística)
covariance = df_raw[['transaction_amount', 'transaction_time']].corr()
print(covariance)
print()

# 2. Separar as 28 features a serem normalizadas
X_features = df_raw.drop(columns=['transaction_time', 'transaction_amount', 'Class'])

# 3. Aplicar normalização
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X_features)

# 4. Renomear para V1 a V28
colunas_v = [f"V{i+1}" for i in range(X_features.shape[1])]
df_v = pd.DataFrame(X_normalized, columns=colunas_v)

# 5. Reorganizar colunas: Time, V1–V28, Amount, Class
df_final = pd.concat([
    df_raw['transaction_time'].rename('Time'),
    df_v,
    df_raw['transaction_amount'].rename('Amount'),
    df_raw['Class']
], axis=1)

# 6. Salvar CSV final
df_final.to_csv("dataset/dataset_normalizado.csv", index=False)

# 7. Visualização
print("✅ Dataset final salvo como 'dataset_normalizado.csv' com colunas na ordem: Time, V1–V28, Amount, Class")
print(df_final.head())
