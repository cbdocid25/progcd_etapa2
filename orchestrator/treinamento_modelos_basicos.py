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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

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
# ETAPA 1: CARREGAR E ESCALAR O DATASET
# ============================================

df = pd.read_csv("../dataset/dataset_com_pca.csv")
scaler = StandardScaler()
df[['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])

# ============================================
# ETAPA 2: DEFINIR VARIÁVEIS
# ============================================

X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# ============================================
# ETAPA 3: TREINAR MODELOS
# ============================================


# Modelo 1: Regressão Logística
# -------------------------------
modelo_lr = LogisticRegression(max_iter=5000, solver='saga')
modelo_lr.fit(X_train, y_train)
y_pred_lr = modelo_lr.predict(X_test)
y_prob_lr = modelo_lr.predict_proba(X_test)[:, 1]


# Modelo 2: Random Forest
# -------------------------------
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_rf.fit(X_train, y_train)
y_pred_rf = modelo_rf.predict(X_test)
y_prob_rf = modelo_rf.predict_proba(X_test)[:, 1]


# Modelo 3: XGBoost
# -------------------------------
modelo_xgb = XGBClassifier(eval_metric='logloss', random_state=42)
modelo_xgb.fit(X_train, y_train)
y_pred_xgb = modelo_xgb.predict(X_test)
y_prob_xgb = modelo_xgb.predict_proba(X_test)[:, 1]

# ============================================
# ETAPA 4: AVALIAR MÉTRICAS
# ============================================

def calcular_metricas(nome, y_true, y_pred, y_prob):
    return {
        "Modelo": nome,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob)
    }

resultados = [
    calcular_metricas("Logistic Regression", y_test, y_pred_lr, y_prob_lr),
    calcular_metricas("Random Forest", y_test, y_pred_rf, y_prob_rf),
    calcular_metricas("XGBoost", y_test, y_pred_xgb, y_prob_xgb)
]

df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv("../dataset/resultados_modelos.csv", index=False)

# ============================================
# ETAPA 5: FINALIZAÇÃO
# ============================================

print("Resultados gerados e salvos com sucesso:")
print(df_resultados)

fim = datetime.now()
duracao = fim - inicio
print(f"Fim da execução: {fim.strftime('%d/%m/%Y %H:%M:%S')}")
print(f"Duração total: {formatar_duracao(duracao)}")
