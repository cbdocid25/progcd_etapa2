# ============================================
# ETAPA 0: IMPORTAR BIBLIOTECAS
# ============================================
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

# ============================================
# Função para formatar a duração
# ============================================
def formatar_duracao(interval):
    t = int(interval.total_seconds())
    return f"{t//3600:02}:{(t%3600)//60:02}:{t%60:02}"

# ============================================
# INÍCIO DA EXECUÇÃO
# ============================================
inicio = datetime.now()
print(f"Início da execução: {inicio.strftime('%d/%m/%Y %H:%M:%S')}")

# ============================================
# ETAPA 1: CARREGAR E NORMALIZAR O DATASET
# ============================================
df = pd.read_csv("../model/dataset/creditcard.csv")

# ✅ Manter Time original — criar Amount_Scaled para os modelos
scaler = StandardScaler()
df['Amount_Scaled'] = scaler.fit_transform(df[['Amount']])

# ============================================
# ETAPA 2: DEFINIR X E y PARA OS MODELOS
# ============================================
# Modelos usarão Amount_Scaled no lugar de Amount
X = df.drop(columns=["Class", "Amount"])
X = X.rename(columns={"Amount_Scaled": "Amount"})
y = df["Class"]

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# ============================================
# ETAPA 3: TREINAMENTO DOS MODELOS
# ============================================
# Regressão Logística
modelo_lr = LogisticRegression(max_iter=5000, solver='saga')
modelo_lr.fit(X_train, y_train)
df['LR_PRED'] = modelo_lr.predict(X)

# Random Forest
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_rf.fit(X_train, y_train)
df['RF_PRED'] = modelo_rf.predict(X)

# XGBoost
modelo_xgb = XGBClassifier(eval_metric='logloss', random_state=42)
modelo_xgb.fit(X_train, y_train)
df['XGB_PRED'] = modelo_xgb.predict(X)

# ============================================
# ETAPA 4: JULGAMENTO POR MODELO
# ============================================
def rotulo_texto(valor):
    return 'FRAUDE' if valor == 1 else 'LEGÍTIMA'

df['LR_RESULT'] = df['LR_PRED'].apply(rotulo_texto)
df['RF_RESULT'] = df['RF_PRED'].apply(rotulo_texto)
df['XGB_RESULT'] = df['XGB_PRED'].apply(rotulo_texto)

# ============================================
# ETAPA 5: SALVAR RELATÓRIO FINAL
# ============================================
# ✅ Remover Amount_Scaled dos dados finais
df_final = df.drop(columns=["Amount_Scaled"])

df_final.to_csv("../dataset/relatorio_treinamento.csv", index=False)
print("✅ Arquivo relatorio_treinamento.csv gerado com sucesso.")

fim = datetime.now()
duracao = fim - inicio
print(f"Fim da execução: {fim.strftime('%d/%m/%Y %H:%M:%S')}")
print(f"Duração total: {formatar_duracao(duracao)}")
