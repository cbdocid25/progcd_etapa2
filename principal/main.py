from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from controllers.engine import executar_codigo, llm, df, code_prompt, explain_prompt, PerguntaHelper
import matplotlib.pyplot as plt
from datetime import datetime
from fpdf import FPDF
from PIL import Image
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

GRAFICO_DIR = "static/graficos"
RELATORIOS_DIR = "relatorios"
os.makedirs(GRAFICO_DIR, exist_ok=True)
os.makedirs(RELATORIOS_DIR, exist_ok=True)

historico = []
grafico_count = 0

@app.get("/apagar")
async def apagar_historico():
    historico.clear()
    return RedirectResponse(url="/", status_code=303)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "historico": historico,
        "resposta": None,
        "pergunta": "",
        "img_path": None
    })

@app.post("/", response_class=HTMLResponse)
async def processar(request: Request, pergunta: str = Form(...)):
    global grafico_count
    try:
        tipo = PerguntaHelper.identificar_tipo_pergunta(pergunta)
        img_path = None
        resultado_para_explicacao = None

        if tipo == "acuracia_modelos":
            resultado = PerguntaHelper.calcular_acuracia_modelos_em_fraudes(df)
            resultado_para_explicacao = resultado
            historico.append((pergunta, None, None))

        elif tipo == "grafico_intervalo":
            h1, h2 = PerguntaHelper.extrair_intervalo_horario(pergunta)
            if h1 is not None and h2 is not None:
                resultado = PerguntaHelper.grafico_dispersao_fraudes_intervalo(df, h1, h2)
                fig_name = os.path.join(GRAFICO_DIR, f"fig_{grafico_count}.png")
                grafico_count += 1
                resultado.set_size_inches(4.4, 2.8)
                resultado.savefig(fig_name, format="png", dpi=100, bbox_inches="tight")
                plt.close(resultado)
                img_path = f"/static/graficos/{os.path.basename(fig_name)}"
                resultado_para_explicacao = "Gráfico gerado com sucesso"
                historico.append((pergunta, None, img_path))

        else:
            codigo = llm.invoke(code_prompt.format(input=pergunta)).content.strip()
            if codigo.startswith("```"):
                codigo = "\n".join([l for l in codigo.splitlines() if not l.strip().startswith("```")])
            resultado = executar_codigo(codigo, df, pergunta)

            if isinstance(resultado, plt.Figure):
                fig_name = os.path.join(GRAFICO_DIR, f"fig_{grafico_count}.png")
                grafico_count += 1
                resultado.set_size_inches(4.4, 2.8)
                resultado.savefig(fig_name, format="png", dpi=100, bbox_inches="tight")
                plt.close(resultado)
                img_path = f"/static/graficos/{os.path.basename(fig_name)}"
                resultado_para_explicacao = "Gráfico gerado com sucesso"
            else:
                resultado_para_explicacao = resultado

            historico.append((pergunta, None, img_path))

        explicacao = llm.invoke(
            explain_prompt.format(pergunta=pergunta, resultado=resultado_para_explicacao)
        ).content.strip()

        historico[-1] = (pergunta, explicacao, img_path)

        return templates.TemplateResponse("index.html", {
            "request": request,
            "historico": historico,
            "resposta": explicacao,
            "pergunta": pergunta,
            "img_path": img_path
        })
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "resposta": f"Erro: {e}",
            "pergunta": pergunta,
            "historico": historico,
            "img_path": None
        })

@app.get("/exportar")
async def exportar_historico():
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(0, 10, "Histórico de Análise de Fraudes com LLM", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, "Este arquivo contém o histórico completo das interações realizadas com o assistente.")
    pdf.ln(5)

    def safe_encode(text):
        try:
            return text.encode("latin-1", "ignore").decode("latin-1")
        except:
            return "[Erro ao codificar caracteres especiais]"

    for idx, (q, r, img_path) in enumerate(historico, 1):
        pdf.set_font("Arial", style='B', size=12)
        pdf.cell(0, 10, f"Pergunta {idx}", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, safe_encode(q))
        pdf.set_font("Arial", style='B', size=12)
        pdf.cell(0, 10, "Resposta", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, safe_encode(r))
        if img_path:
            try:
                img_local = os.path.join("static/graficos", os.path.basename(img_path))
                im = Image.open(img_local).convert("RGB")
                temp_path = img_local.replace(".png", "_temp.jpg")
                im.save(temp_path, format="JPEG")
                pdf.image(temp_path, w=170)
            except Exception as e:
                pdf.multi_cell(0, 10, f"[Erro ao carregar imagem: {img_path}]")
        pdf.ln(5)

    export_path = os.path.join(RELATORIOS_DIR, f"Historico_LLM_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    pdf.output(export_path)
    return FileResponse(export_path, media_type='application/pdf', filename=os.path.basename(export_path))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=5552, reload=True)

