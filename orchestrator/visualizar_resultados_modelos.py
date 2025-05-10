# ============================================
# SCRIPT: visualizar_resultados_modelos.py
# Gera e salva gráfico de barras das métricas
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Carregar os resultados dos modelos
df_resultados = pd.read_csv("../dataset/resultados_modelos.csv")

# 2. Definir métricas a serem plotadas
metricas = ['F1-Score', 'Precision', 'Recall', 'ROC-AUC']

# 3. Criar pasta de saída (se não existir)
output_dir = "../dataset/imagens"
os.makedirs(output_dir, exist_ok=True)

# 4. Gerar e salvar gráficos
for metrica in metricas:
    plt.figure(figsize=(7, 4))
    plt.bar(df_resultados['Modelo'], df_resultados[metrica], color='skyblue')
    plt.title(f"Comparação de {metrica} entre os modelos")
    plt.ylim(0, 1)
    plt.ylabel(metrica)
    plt.xlabel("Modelos")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Salvar imagem
    caminho_arquivo = os.path.join(output_dir, f"grafico_{metrica.lower().replace('-', '_')}.png")
    plt.savefig(caminho_arquivo)
    plt.show()

print("✅ Gráficos salvos em: ../dataset/imagens/")
