import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

"""
O que é esse código?
É um buscador semântico de desempenho de modelos de machine learning, 
que usa embeddings (SentenceTransformer) e FAISS para permitir consultas 
em linguagem natural sobre um dataset de métricas.
"""

# 1. Carregar o CSV correto com os resultados dos modelos
df = pd.read_csv("../metricas/dataset/metricas_treinamento.csv")

# 2. Gerar frases explicativas para cada modelo
frases = []
for _, row in df.iterrows():
    modelo = row["Modelo"]
    frases.append(f"O modelo {modelo} teve precisão de {row['Precision']:.2f}")
    frases.append(f"O modelo {modelo} teve recall de {row['Recall']:.2f}")
    frases.append(f"O modelo {modelo} teve F1-Score de {row['F1-Score']:.2f}")
    frases.append(f"O modelo {modelo} teve ROC-AUC de {row['ROC-AUC']:.2f}")

# 3. Criar embeddings
modelo_embedding = SentenceTransformer("all-MiniLM-L6-v2")
vetores = modelo_embedding.encode(frases)

# 4. Indexar com FAISS
dim = vetores.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(vetores))

# 5. Fazer consulta semântica
consulta = "Qual modelo teve melhor F1?"
vetor_consulta = modelo_embedding.encode([consulta])
distancias, indices = index.search(np.array(vetor_consulta), k=10)

# 6. Exibir resultados
print("🔍 Resultados mais próximos da consulta:")
for i in indices[0]:
    print("-", frases[i])
