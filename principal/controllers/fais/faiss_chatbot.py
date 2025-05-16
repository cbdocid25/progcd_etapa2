
# ============================================
# ETAPA 0: IMPORTAR BIBLIOTECAS
# ============================================
import pandas as pd
from sentence_transformers import SentenceTransformer
from modulo_perguntas import gerar_frases_resumo
import faiss
import numpy as np

# ============================================
# ETAPA 1: LER O RELATÓRIO DE TREINAMENTO E O ORIGINAL
# ============================================
df = pd.read_csv("../../dataset/relatorio_treinamento.csv")
df_original = pd.read_csv("../../model/dataset/creditcard.csv")
df["Amount_Real"] = df_original["Amount"]

# Converter a coluna 'Time' normalizada em hora do dia
df['hora'] = (df['Time'] * 172800 / 3600).round() % 24

# ============================================
# ETAPA 2: GERAR INSIGHTS DO DATASET
# ============================================
frases = gerar_frases_resumo(df)

# ============================================
# ETAPA 3: VISUALIZAR TODAS AS FRASES INDEXADAS
# ============================================
print("\n📌 Frases indexadas para busca semântica:")
for frase in frases:
    print("-", frase)

# ============================================
# ETAPA 4: VETORIZAÇÃO COM EMBEDDINGS
# ============================================
modelo = SentenceTransformer("all-MiniLM-L6-v2")
vetores = modelo.encode(frases)

# ============================================
# ETAPA 5: INDEXAÇÃO FAISS
# ============================================
dim = vetores.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(vetores))

# ============================================
# ETAPA 6: CONSULTA INTERATIVA
# ============================================

while True:
    consulta = input("\n🔎 Digite sua pergunta: \n")
    if consulta == "S":
        print("Encerrando...")
        break

    vetor_consulta = modelo.encode([consulta])
    _, indices = index.search(np.array(vetor_consulta), k=2)

    print("\n✅ Respostas mais relevantes: 'S' para sair")
    for i in indices[0]:
        print("-", frases[i])


