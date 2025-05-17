import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from controllers.helper import PerguntaHelper

# 1. Carregar variáveis de ambiente
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("A variável OPENAI_API_KEY não foi definida.")

# 2. Carregar dataset
DATASET_PATH = "model/dataset/relatorio_treinamento.csv"
df = pd.read_csv(DATASET_PATH)
print(f"CSV carregado com {len(df)} registros.")

# 3. Instanciar LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# 4. Prompts
code_prompt = ChatPromptTemplate.from_template("""
Você é um assistente Python. Um DataFrame chamado `df` contém dados de transações com cartões de crédito.
Colunas disponíveis: 'Hora', 'Amount (€)', 'Class', 'LR_PRED', 'RF_PRED', 'XGB_PRED', 'LR_RESULT', 'RF_RESULT', 'XGB_RESULT', 'V1' a 'V28'.

Use a coluna 'Hora' para representar o tempo (em horas do dia) e a coluna 'Amount (€)' como valor monetário.

Retorne apenas o código Python (sem explicação) que responde a esta pergunta. A resposta final deve ser atribuída à variável `resultado`.

Pergunta: {input}
""")

explain_prompt = ChatPromptTemplate.from_template("""
Baseado na pergunta: "{pergunta}"
E no resultado: "{resultado}"

Gere uma resposta clara, explicativa e amigável para humanos. Use linguagem natural e inclua contexto.
""")

# 5. Execução segura de código

def executar_codigo(codigo, df, pergunta):
    df_copia = df.copy()
    df_copia["Hora"] = (df_copia["Time"] // 3600 % 24).astype(int)
    df_copia["Amount (€)"] = df_copia["Amount"]
    local_vars = {"df": df_copia}

    try:
        exec(codigo, {}, local_vars)
        saida = local_vars.get("resultado")

        if isinstance(saida, pd.DataFrame):
            saida = PerguntaHelper.aplicar_formatacoes_em_dataframe(saida)

        if plt.get_fignums():
            fig = plt.gcf()
            return fig

        if isinstance(saida, (int, float)) and "hora" in pergunta.lower():
            saida = PerguntaHelper.converter_time_para_hora_minuto(saida)
        elif isinstance(saida, (int, float)) and ("amount" in pergunta.lower() or "valor" in pergunta.lower()):
            saida = PerguntaHelper.formatar_como_euro(saida)

        saida = PerguntaHelper.formatar_time_em_dict(saida)
        return saida

    except Exception as e:
        return f"Erro ao executar: {e}"

# 6. Função de entrada principal da aplicação
def processar_pergunta(pergunta):
    tipo = PerguntaHelper.identificar_tipo_pergunta(pergunta)
    img_path = None
    resultado_para_explicacao = None

    if tipo == "acuracia_modelos":
        resultado = PerguntaHelper.calcular_acuracia_modelos_em_fraudes(df)
        resultado_para_explicacao = resultado

    elif tipo == "grafico_intervalo":
        h1, h2 = PerguntaHelper.extrair_intervalo_horario(pergunta)
        if h1 is not None and h2 is not None:
            resultado = PerguntaHelper.grafico_dispersao_fraudes_intervalo(df, h1, h2)
            from uuid import uuid4
            nome = f"fig_{uuid4().hex[:8]}.png"
            fig_path = os.path.join("static/graficos", nome)
            resultado.set_size_inches(4.4, 2.8)
            resultado.savefig(fig_path, format="png", dpi=100, bbox_inches="tight")
            plt.close(resultado)
            img_path = f"/static/graficos/{nome}"
            resultado_para_explicacao = "Gráfico gerado com sucesso"

    else:
        codigo = llm.invoke(code_prompt.format(input=pergunta)).content.strip()
        if codigo.startswith("```"):
            codigo = "\n".join([l for l in codigo.splitlines() if not l.strip().startswith("```")])
        resultado = executar_codigo(codigo, df, pergunta)

        if isinstance(resultado, plt.Figure):
            from uuid import uuid4
            nome = f"fig_{uuid4().hex[:8]}.png"
            fig_path = os.path.join("static/graficos", nome)
            resultado.set_size_inches(4.4, 2.8)
            resultado.savefig(fig_path, format="png", dpi=100, bbox_inches="tight")
            plt.close(resultado)
            img_path = f"/static/graficos/{nome}"
            resultado_para_explicacao = "Gráfico gerado com sucesso"
        else:
            resultado_para_explicacao = resultado

    explicacao = llm.invoke(explain_prompt.format(pergunta=pergunta, resultado=resultado_para_explicacao)).content.strip()

    return explicacao, img_path
