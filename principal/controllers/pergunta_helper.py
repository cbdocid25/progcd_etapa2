import locale
import re

class PerguntaHelper:
    @staticmethod
    def converter_time_para_hora_minuto(segundos):
        hora = int(segundos // 3600)
        minuto = int((segundos % 3600) // 60)
        return f"{hora:02d}:{minuto:02d}h"

    @staticmethod
    def formatar_time_em_dict(dados):
        def formatar(linha):
            if "Time" in linha and isinstance(linha["Time"], (int, float)):
                linha["Time"] = PerguntaHelper.converter_time_para_hora_minuto(linha["Time"])
            return linha

        if isinstance(dados, dict):
            return formatar(dados)
        if isinstance(dados, list) and all(isinstance(item, dict) for item in dados):
            return [formatar(item) for item in dados]
        return dados

    @staticmethod
    def formatar_como_euro(valor):
        try:
            valor_formatado = f"€{valor:,.2f}"
            valor_formatado = valor_formatado.replace(",", "X").replace(".", ",").replace("X", ".")
            return valor_formatado
        except:
            return f"€{valor}"

    @staticmethod
    def calcular_acuracia_modelos_em_fraudes(df):
        fraudes = df[df['Class'] == 1]
        resultado = {
            "LR": round((fraudes["LR_RESULT"] == "FRAUDE").mean() * 100, 2),
            "RF": round((fraudes["RF_RESULT"] == "FRAUDE").mean() * 100, 2),
            "XGB": round((fraudes["XGB_RESULT"] == "FRAUDE").mean() * 100, 2)
        }
        return resultado

    @staticmethod
    def grafico_dispersao_fraudes_intervalo(df, hora_inicio=0, hora_fim=7):
        import matplotlib
        matplotlib.use('Agg')  # <- Impede a abertura de janela
        import matplotlib.pyplot as plt

        start_sec = hora_inicio * 3600
        end_sec = hora_fim * 3600
        fraudes = df[df['Class'] == 1]
        intervalo = fraudes[(fraudes['Time'] >= start_sec) & (fraudes['Time'] <= end_sec)]
        fig, ax = plt.subplots()
        ax.scatter(intervalo['Time'], intervalo['Amount'], alpha=0.6)
        ax.set_title(f"Fraudes entre {hora_inicio:02d}:00 e {hora_fim:02d}:00")
        ax.set_xlabel("Horário da Transação (segundos desde 00:00)")
        ax.set_ylabel("Valor da Transação (€)")
        return fig

    @staticmethod
    def extrair_intervalo_horario(pergunta):
        pergunta = pergunta.lower()
        padrao = r"(\d{1,2})[:h]{0,2}\s*(?:até|as|a|e)\s*(\d{1,2})[:h]{0,2}"
        match = re.search(padrao, pergunta)
        if match:
            h1 = int(match.group(1))
            h2 = int(match.group(2))
            return min(h1, h2), max(h1, h2)
        return None, None

    @staticmethod
    def identificar_tipo_pergunta(pergunta: str) -> str:
        pergunta = pergunta.lower()
        if "fraude" in pergunta and re.search(r"\d{1,2}[:h]{0,2}.*(?:até|as|a|e).*\d{1,2}[:h]{0,2}", pergunta):
            return "grafico_intervalo"
        if any(p in pergunta for p in ["modelo", "lr", "rf", "xgb"]):
            return "acuracia_modelos"
        return "padrao"
