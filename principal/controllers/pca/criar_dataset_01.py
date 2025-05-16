import pandas as pd
import numpy as np

# Garantir reprodutibilidade
np.random.seed(42)

# Lista com exatamente 28 colunas de features simuladas (SEM time e amount)
feature_names = [
    'account_balance', 'customer_age', 'merchant_rating', 'days_since_last_tx',
    'transaction_hour', 'customer_tenure', 'num_prev_frauds', 'prev_chargebacks',
    'transaction_frequency', 'avg_monthly_spend', 'device_trust_score', 'merchant_transaction_count',
    'customer_satisfaction', 'payment_installments', 'location_risk_score', 'device_age_days',
    'login_attempts', 'previous_denials', 'card_swipe_speed', 'merchant_response_time',
    'customer_income', 'available_credit', 'merchant_discount_rate', 'delivery_time_estimate',
    'loyalty_score', 'bonus_points_used', 'merchant_id_score', 'risk_flag_score'
]

# Gerar os dados das features
data_raw = {}
for name in feature_names:
    if "balance" in name or "income" in name or "spend" in name or "credit" in name:
        data_raw[name] = np.random.uniform(100.0, 10000.0, size=20)
    elif "rating" in name or "score" in name:
        data_raw[name] = np.random.uniform(0.0, 1.0, size=20)
    elif "frequency" in name or "count" in name or "attempts" in name:
        data_raw[name] = np.random.randint(0, 100, size=20)
    elif "points" in name:
        data_raw[name] = np.random.randint(0, 1000, size=20)
    elif "age" in name or "days" in name or "tenure" in name:
        data_raw[name] = np.random.randint(0, 3650, size=20)
    elif "hour" in name:
        data_raw[name] = np.random.randint(0, 24, size=20)
    elif "installments" in name or "denials" in name or "frauds" in name:
        data_raw[name] = np.random.randint(0, 5, size=20)
    else:
        data_raw[name] = np.random.randint(0, 10, size=20)

# Gerar as colunas separadas: transaction_time, transaction_amount e Class
data_raw['transaction_time'] = np.random.randint(0, 86400, size=20)
data_raw['transaction_amount'] = np.random.uniform(1.0, 2000.0, size=20)
data_raw['Class'] = np.random.choice([0, 1], size=20, p=[0.95, 0.05])

# Criar o DataFrame
df_raw = pd.DataFrame(data_raw)

# Exportar para CSV
df_raw.to_csv("dataset/dataset_simulado.csv", index=False)

print("✅ Tabela crua simulada gerada com sucesso com 28 features + transaction_time + transaction_amount + Class!")
print(f"Total de colunas: {df_raw.shape[1]}")
print(f"Colunas: {df_raw.columns.tolist()}")
