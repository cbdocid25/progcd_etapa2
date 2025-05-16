import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Evita janelas de exibição em ambientes web/CLI
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from controllers.pergunta_helper import PerguntaHelper

# 1. Carregar variáveis de ambiente
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("A variável OPENAI_API_KEY não foi definida.")

# 2. Carregar dataset
data_path = "model/dataset/relatorio_treinamento.csv"
df = pd.read_csv(data_path)
print(f"CSV carregado com {len(df)} registros.")

# 3. Instanciar LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# 4. Prompts
code_prompt = ChatPromptTemplate.from_template("""
Você é um assistente Python. Um DataFrame chamado `df` contém dados de transações com cartões de crédito.
Colunas disponíveis: 'Time', 'Amount', 'Class', 'LR_PRED', 'RF_PRED', 'XGB_PRED', 'LR_RESULT', 'RF_RESULT', 'XGB_RESULT', 'V1' a 'V28'.

Retorne apenas o código Python (sem explicação) que responde a esta pergunta. A resposta final deve ser atribuída à variável `resultado`.

Pergunta: {input}
""")

explain_prompt = ChatPromptTemplate.from_template("""
Baseado na pergunta: "{pergunta}"
E no resultado: "{resultado}"

Gere uma resposta clara, explicativa e amigável para humanos. Use linguagem natural e inclua contexto.
""")

# 5. Execução segura
def executar_codigo(codigo, df, pergunta):
    local_vars = {"df": df.copy()}
    try:
        exec(codigo, {}, local_vars)
        saida = local_vars.get("resultado")

        if plt.get_fignums():
            fig = plt.gcf()
            return fig

        if isinstance(saida, (int, float)) and "hora horario horário time " in pergunta.lower():
            saida = PerguntaHelper.converter_time_para_hora_minuto(saida)
        elif isinstance(saida, (int, float)) and ("amount" in pergunta.lower() or "valor" in pergunta.lower()):
            saida = PerguntaHelper.formatar_como_euro(saida)

        saida = PerguntaHelper.formatar_time_em_dict(saida)
        return saida

    except Exception as e:
        return f"Erro ao executar: {e}"

# 6. Interação direta via terminal
if __name__ == "__main__":
    print("\nSistema pronto com respostas explicadas por LLM.\n")

    while True:
        pergunta = input("Sua pergunta: ").strip()
        if pergunta.lower() == "sair":
            print("Encerrando.")
            break

        try:
            tipo = PerguntaHelper.identificar_tipo_pergunta(pergunta)

            if tipo == "acuracia_modelos":
                resultado = PerguntaHelper.calcular_acuracia_modelos_em_fraudes(df)

            elif tipo == "grafico_intervalo":
                h1, h2 = PerguntaHelper.extrair_intervalo_horario(pergunta)
                if h1 is not None and h2 is not None:
                    resultado = PerguntaHelper.grafico_dispersao_fraudes_intervalo(df, h1, h2)
                    print(f"\nGráfico gerado para fraudes entre {h1:02d}:00 e {h2:02d}:00.")
                    print("Gráfico pronto (não exibido no terminal).")
                    continue
                else:
                    resultado = "Não foi possível identificar o intervalo de horário."

            else:
                codigo = llm.invoke(code_prompt.format(input=pergunta)).content.strip()
                if codigo.startswith("```"):
                    codigo = "\n".join([linha for linha in codigo.splitlines() if not linha.strip().startswith("```")])
                print(f"\nCódigo gerado:\n{codigo}\n")
                resultado = executar_codigo(codigo, df, pergunta)

            explicacao = llm.invoke(explain_prompt.format(pergunta=pergunta, resultado=resultado)).content.strip()
            print("\nResposta com explicação:")
            print(explicacao)

        except Exception as e:
            print(f"Erro: {e}")
