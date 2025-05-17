# Detecção de Fraudes em Cartões de Crédito com LLM

Este projeto integra técnicas de aprendizado de máquina e modelos de linguagem (LLMs) com a abordagem RAG (Retrieval-Augmented Generation) para detecção e explicação de fraudes em cartões de crédito.

## Objetivo

Desenvolver uma solução capaz de identificar transações fraudulentas e, ao mesmo tempo, explicar em linguagem natural as decisões do modelo, utilizando uma LLM integrada via LangChain.

## Como Executar o Aplicativo de Interação com LLM

Este projeto permite a interação com um modelo de linguagem (LLM) para análise de fraudes em transações de cartão de crédito. Ele foi desenvolvido com o objetivo de tornar a exploração de dados mais intuitiva, através de perguntas em linguagem natural.

---

## Estrutura Principal

O projeto é composto por três arquivos principais:

- `helper.py`: contém funções auxiliares para tratamento de dados e formatação de respostas.  
- `engine.py`: orquestra a lógica de consulta, integração com a LLM e formatação dos resultados.  
- `main.py`: ponto de entrada da aplicação. Ao ser executado, inicia a interface e permite interação com o modelo.

---

## Executando a Aplicação

Certifique-se de que todas as dependências estão instaladas (ver seção abaixo).
No terminal, execute o seguinte comando:

```bash
python main.py
```

A aplicação será iniciada.
Você poderá fazer perguntas como:
- "Quantas transações fraudulentas existem no dataset?"
- "Qual foi o maior valor entre as fraudes?"
- "Crie um gráfico de dispersão entre fraudes e operações legítimas e atribua as cores azul para legítima e laranja para as fraudes"

O sistema interpretará a pergunta e apresentará a resposta com base no dataset de fraude de cartões de crédito.

## Requisitos

Antes de rodar, instale as dependências com:
```bash
pip install -r requirements.txt
```

Se estiver utilizando ambiente virtual (recomendado), ative-o antes de executar o comando acima.

---

## Observação
Este projeto utiliza uma arquitetura modular com suporte a agentes LLM, oferecendo uma camada inteligente sobre os dados para facilitar análises sem necessidade de escrever código ou queries complexas.


## Dataset

O dataset utilizado neste projeto é baseado no conjunto de dados original do Kaggle sobre detecção de fraudes em cartões de crédito. Este arquivo foi **manipulado e enriquecido** com os julgamentos de **três modelos de aprendizado supervisionado** (Regressão Logística, Random Forest e XGBoost), resultando em colunas adicionais com as predições de cada modelo.

Devido ao seu tamanho final (153.9MB), ele não pôde ser hospedado diretamente neste repositório GitHub.
Você pode baixá-lo através do link abaixo:

🔗 [Clique aqui para acessar o dataset no Google Drive](https://drive.google.com/drive/folders/1Gp9VLrw08iapNNpInsbsl6o9w46Jb9bT?usp=drive_link)

Após o download, adicione o arquivo na seguinte estrutura de diretório dentro do projeto:

```
nome_do_projeto/
└── model/
    └── dataset/
        └── relatorio_treinamento.csv
```

> **Importante:** o caminho acima deve ser seguido exatamente para garantir o funcionamento do sistema, conforme definido na variável:

```python
DATASET_PATH = "model/dataset/relatorio_treinamento.csv"
```

## Fonte do dataset original:
🔗 [Click aqui para acessar o dataset no Kaggle](https://www.kaggle.com/code/gpreda/credit-card-fraud-detection-predictive-models)


## Tecnologias e Ferramentas

- Python 3.10+
- Jupyter Notebook
- Pandas, Scikit-Learn, XGBoost
- SHAP (para interpretabilidade)
- FAISS (para criação de vetores)
- LangChain (para integração com LLM)
- OpenAI API (ou outra LLM compatível)

## Abordagem

1. Pré-processamento do dataset.
2. Treinamento de modelos de detecção (baseline).
3. Explicabilidade com SHAP values.
4. Indexação vetorial com FAISS.
5. Integração com LangChain para geração de explicações usando LLM.
6. Comparação dos modelos com e sem auxílio da LLM.

## Equipe

- Alexandre Teixeira da Silva
- César Braz de Oliveira
- Ícaro Guimarães Canto
- Priscila Leylianne da Silva Gonçalves

# Etapa 1.3 — Aplicar Pré-processamento: Normalização + PCA

Nesta etapa, realizamos o pré-processamento do dataset original de detecção de fraudes em transações de cartão de crédito, aplicando duas técnicas fundamentais:

---

## Objetivos

- Normalizar as variáveis contínuas do dataset (`Time` e `Amount`)
- Aplicar PCA (Análise de Componentes Principais) para reduzir a dimensionalidade das 28 features anonimizadas (`V1` a `V28`)
- Gerar um novo dataset reduzido, facilitando a visualização, detecção de padrões e aplicação de modelos futuros

---

## Técnicas aplicadas

### 1. Normalização (StandardScaler)
- Utilizamos a técnica de padronização Z-score (`StandardScaler`) para transformar `Time` e `Amount` em dados com média 0 e desvio padrão 1
- Isso evita que essas colunas interfiram negativamente nos algoritmos baseados em distância ou gradiente

### 2. PCA — Principal Component Analysis
- Redução das 28 colunas (`V1` a `V28`) para os 10 principais componentes (`PCA1` a `PCA10`)
- Retenção de mais de **80% da variância total** do dataset original
- Aplicação com `sklearn.decomposition.PCA(n_components=10)`

---

## Arquivos envolvidos

- **Entrada**: `dataset_simulado.csv` ou `creditcard.csv` (versão original)
- **Saída 1**: `dataset_normalizado.csv` (com `Time`, `V1`–`V28`, `Amount`, `Class` normalizados)
- **Saída 2**: `dataset_com_pca.csv` (com `PCA1` a `PCA10`, `Time`, `Amount`, `Class`)

---

## Por que aplicar PCA?

- Reduz o custo computacional em modelos futuros
- Remove multicolinearidade entre variáveis
- Facilita visualizações 2D e clustering
- Preserva o máximo de informação com menos colunas

---

## Como executar esta etapa

```bash
python pre_processamento_pca.py
```

Certifique-se de ter as bibliotecas instaladas:

```bash
pip install pandas scikit-learn
```

---

## Resultado Esperado

- Dataset escalonado
- Dataset reduzido com 10 colunas PCA
- CSVs prontos para serem usados no treinamento de modelos (Etapa 2.1)

---
# Etapa 2.1 — Treinar Modelos Supervisionados

Nesta etapa do projeto, realizamos o treinamento de modelos supervisionados utilizando os dados pré-processados do dataset de transações financeiras para detecção de fraudes.

---

## Objetivo

- Utilizar algoritmos de classificação supervisionada para prever a classe da transação (`0` = legítima, `1` = fraude)
- Avaliar a performance dos modelos com base em métricas padrão
- Salvar os resultados comparativos para futuras análises e visualizações

---

## Arquivo de entrada

- `creditcard.csv` (versão original do Kaggle com `Time`, `V1–V28`, `Amount`, `Class`)
- Pré-requisito: `Time` e `Amount` devem estar normalizados com `StandardScaler`

---

## Modelos treinados

### 1. Logistic Regression
- Algoritmo linear que usa função sigmoide para prever a probabilidade de fraude
- Otimizador: `saga`
- Iterações: até `5000`

### 2. Random Forest
- Conjunto de árvores de decisão aleatórias com votação
- Robusto e eficaz mesmo com dados desbalanceados
- `n_estimators = 100`

---

## Métricas geradas

Para cada modelo, foram calculadas as seguintes métricas:

| Métrica     | Descrição                                           |
|-------------|------------------------------------------------------|
| Accuracy    | Proporção de acertos sobre o total                   |
| Precision   | Proporção de fraudes corretas entre as previstas     |
| Recall      | Proporção de fraudes encontradas entre as reais      |
| F1-Score    | Média harmônica entre Precision e Recall             |
| ROC-AUC     | Capacidade do modelo em distinguir classes           |

---

## Resultado salvo

- Os resultados das métricas foram organizados em um arquivo:
  - `resultados_modelos.csv`

Exemplo (valores ilustrativos):

| Modelo              | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.9992   | 0.76      | 0.65   | 0.70     | 0.95    |
| Random Forest       | 0.9996   | 0.89      | 0.79   | 0.84     | 0.98    |

---

## Execução

```bash
python treinamento_modelos_basicos.py
```

Bibliotecas necessárias:

```bash
pip install pandas scikit-learn
```

---

## Saída

- `resultados_modelos.csv`: comparativo dos dois algoritmos
- Impressão dos tempos de execução:
  - Início
  - Fim
  - Duração total

---
