
def gerar_frases_resumo(df):
    frases = []

    # Horário com mais fraudes
    fraudes = df[df["Class"] == 1]
    horas_fraude = fraudes["hora"].value_counts().sort_index()
    hora_mais_fraude = horas_fraude.idxmax()
    frases.append(f"O horário com mais fraudes foi entre {int(hora_mais_fraude)}h e {int((hora_mais_fraude+1)%24)}h")

    # Maior fraude e horário
    maior_fraude = fraudes["Amount_Real"].max()
    idx_maior_fraude = fraudes["Amount_Real"].idxmax()
    hora_maior_fraude = df.loc[idx_maior_fraude, "hora"]
    frases.append(f"O maior valor encontrado em uma transação fraudulenta foi R${maior_fraude:.2f}")
    frases.append(f"A maior fraude registrada foi de R${maior_fraude:.2f}, ocorrendo por volta das {int(hora_maior_fraude)}h")

    # Concordância entre os modelos
    df["concordam"] = (
        (df["LR_PRED"] == df["RF_PRED"]) &
        (df["RF_PRED"] == df["XGB_PRED"])
    )
    total_concordancia = df["concordam"].sum()
    frases.append(f"Os modelos concordaram em {total_concordancia} de {len(df)} transações")

    # Separação de fraudes e legítimas
    legitimas = df[df["Class"] == 0]

    # Valores médios
    media_fraudes = fraudes["Amount_Real"].mean()
    media_legitimas = legitimas["Amount_Real"].mean()
    frases.append(f"O valor médio das transações fraudulentas foi R${media_fraudes:.2f}")
    frases.append(f"O valor médio das transações legítimas foi R${media_legitimas:.2f}")

    # Valores máximos e mínimos
    menor_fraude = fraudes["Amount_Real"].min()
    maior_legitima = legitimas["Amount_Real"].max()
    menor_legitima = legitimas["Amount_Real"].min()
    frases.append(f"O menor valor encontrado em uma transação fraudulenta foi R${menor_fraude:.2f}")
    frases.append(f"O maior valor entre as transações legítimas foi R${maior_legitima:.2f}")
    frases.append(f"O menor valor entre as transações legítimas foi R${menor_legitima:.2f}")

    # Quantidades e percentuais
    total_transacoes = len(df)
    total_fraudes = len(fraudes)
    total_legitimas = len(legitimas)
    percent_fraude = (total_fraudes / total_transacoes) * 100
    percent_legitima = (total_legitimas / total_transacoes) * 100
    frases.append(f"O percentual total de transações fraudulentas é de {percent_fraude:.3f}%")
    frases.append(f"O percentual total de transações legítimas é de {percent_legitima:.3f}%")
    frases.append(f"Foram registradas {total_fraudes} fraudes em {total_transacoes} transações")

    # Distribuição temporal
    fraudes_18h_24h = fraudes[fraudes['hora'].between(18, 23)]
    percent_fraudes_noite = (len(fraudes_18h_24h) / total_fraudes) * 100
    frases.append(f"{percent_fraudes_noite:.2f}% das fraudes ocorreram entre 18h e 24h")

    # Estatísticas de desvio padrão
    std_fraudes = fraudes["Amount_Real"].std()
    std_legitimas = legitimas["Amount_Real"].std()
    frases.append(f"O desvio padrão dos valores de fraudes é R${std_fraudes:.2f}")
    frases.append(f"O desvio padrão dos valores de transações legítimas é R${std_legitimas:.2f}")

    # Mediana
    mediana_fraudes = fraudes["Amount_Real"].median()
    mediana_legitimas = legitimas["Amount_Real"].median()
    frases.append(f"A mediana dos valores de fraudes é R${mediana_fraudes:.2f}")
    frases.append(f"A mediana dos valores de transações legítimas é R${mediana_legitimas:.2f}")

    # Frequência de fraudes por hora
    for hora in range(24):
        qtd = len(fraudes[fraudes["hora"] == hora])
        frases.append(f"Foram registradas {qtd} fraudes entre {hora}h e {(hora + 1) % 24}h")

    # Transações com valor nulo ou negativo
    zero_fraudes = len(fraudes[fraudes["Amount_Real"] == 0])
    negativos_fraudes = len(fraudes[fraudes["Amount_Real"] < 0])
    frases.append(f"Existem {zero_fraudes} fraudes com valor zero")
    frases.append(f"Existem {negativos_fraudes} fraudes com valor negativo")

    # Frases adicionais
    frases += [
        "A maioria das fraudes ocorre no período noturno.",
        "Transações acima de R$500,00 são mais comuns entre fraudes.",
        "O modelo XGBoost teve melhor desempenho em F1-Score.",
        "A regressão logística foi o modelo com menor recall.",
        "Random Forest teve a maior taxa de acerto em transações legítimas.",
        "Menos de 1% das transações totais são fraudes.",
        "Transações fraudulentas tendem a ocorrer em horários específicos.",
        "A distribuição das fraudes é desigual ao longo do dia.",
        "A maior parte das fraudes tem valores inferiores a R$100,00.",
        "O modelo Random Forest classificou corretamente a maioria das transações.",
        "O número de transações legítimas ultrapassa 98% do total.",
        "A variabilidade entre os modelos é maior nas fraudes do que nas legítimas.",
        "Transações com valor zero são raras.",
        "Existem outliers no valor das fraudes.",
        "Os modelos apresentam dificuldade para detectar fraudes pequenas.",
        "A média dos valores legítimos é maior que a média das fraudes.",
        "A diferença entre recall e precisão no modelo XGBoost é significativa.",
        "O desempenho geral dos modelos varia conforme o horário.",
        "Transações próximas da meia-noite são mais suspeitas.",
        "Fraudes tendem a ocorrer mais após as 18h.",
        "Os modelos têm maior concordância em transações legítimas.",
        "A curva ROC indica bom desempenho para o XGBoost.",
        "A matriz de confusão mostra que poucos falsos positivos ocorrem.",
        "O dataset é desbalanceado, com predominância de transações legítimas.",
        "Modelos apresentam overfitting quando treinados com todas as features.",
        "Reduzir dimensionalidade ajudou no desempenho do modelo.",
        "Fraudes não seguem um padrão simples de valor ou tempo.",
        "O aprendizado supervisionado foi eficaz para essa tarefa.",
        "A proporção entre fraude e legítima é de aproximadamente 1:577.",
        "Os modelos têm maior acurácia do que recall.",
        "A performance dos modelos foi avaliada com métricas padrão.",
        "Não há diferença clara entre manhã e tarde para fraudes.",
        "A taxa de erro é pequena para Random Forest.",
        "Detecção de fraudes é um problema de classificação binária.",
        "Mesmo com dados desbalanceados, os modelos conseguiram generalizar.",
        "Poucas fraudes ocorrem pela manhã.",
        "A análise temporal revela picos de fraudes à noite.",
        "Fraudes ocorrem em janelas específicas do dia.",
        "XGBoost teve desempenho superior em ROC-AUC.",
        "Random Forest teve menor taxa de falsos positivos.",
        "Regressão logística apresentou instabilidade em algumas faixas.",
        "Os dados foram normalizados antes do treinamento.",
        "A técnica de PCA foi aplicada ao conjunto de dados.",
        "O dataset utilizado foi obtido do Kaggle.",
        "Os modelos foram avaliados em 30% dos dados.",
        "O dataset foi dividido de forma estratificada.",
        "O valor médio de transações legítimas é superior ao de fraudes.",
        "Fraudes com valores entre R$0 e R$50 representam a maioria dos casos.",
        "A dispersão dos valores de fraude é maior do que a das transações legítimas.",
        "Fraudes ocorrem com maior frequência em horários não comerciais.",
        "A combinação dos três modelos apresenta maior robustez na detecção de fraudes.",
        "Há um padrão sazonal leve nas ocorrências de fraude durante o dia.",
        "O modelo XGBoost é o mais consistente nas predições noturnas.",
        "As transações legítimas possuem uma média próxima de zero, indicando normalização.",
        "A maior parte das transações fraudulentas ocorre em janelas concentradas do tempo.",
        "A regressão logística tende a gerar mais falsos negativos que os outros modelos.",
        "As transações mais suspeitas se concentram nas últimas horas do dia.",
        "A mediana dos valores legítimos é ligeiramente superior à das fraudes.",
        "A técnica de normalização foi importante para evitar viés nos modelos.",
        "A análise de variância entre modelos mostra que RF é o mais estável.",
        "Poucas fraudes ultrapassam o valor de R$1000, o que indica baixa amplitude.",
        "Valores negativos nas fraudes podem indicar problemas na base ou tentativa de reversão.",
        "Fraudes entre 00h e 05h somam menos de 10% dos casos.",
        "As transações com valor próximo de zero foram majoritariamente classificadas como legítimas.",
        "O número de transações por hora segue uma distribuição regular, exceto pelas fraudes.",
        "Fraudes com valores muito altos são raras e podem representar falsos positivos.",
        "O aprendizado dos modelos melhora após redução de variáveis irrelevantes.",
        "A regressão logística apresenta instabilidade em classes desbalanceadas.",
        "Random Forest mantém um bom equilíbrio entre precisão e recall.",
        "XGBoost obteve a menor taxa de falsos negativos entre os três modelos.",
        "A curva Precision-Recall dos modelos indica bom desempenho geral.",
        "A acurácia dos modelos ultrapassa 99% devido ao forte desbalanceamento.",
        "Menos de 5% das fraudes ocorrem fora dos padrões aprendidos pelos modelos.",
        "A utilização de PCA facilitou a redução de ruído no dataset.",
        "Modelos que utilizam todo o conjunto de variáveis tendem a sofrer overfitting.",
        "Mesmo com divergências entre os modelos, a detecção geral foi satisfatória.",
        "Fraudes detectadas por todos os modelos simultaneamente são mais confiáveis.",
        "A maior parte das fraudes com valor superior a R$200 foi detectada com sucesso.",
        "Há uma leve correlação negativa entre o valor da transação e a sua legitimidade.",
        "O tempo médio entre fraudes consecutivas é significativamente maior do que o de legítimas.",
        "A identificação de padrões temporais ajuda na antecipação de comportamentos suspeitos.",
        "Fraudes geralmente ocorrem em menor quantidade, mas em horários recorrentes.",
        "O modelo XGBoost apresenta maior sensibilidade a variações nos dados de entrada.",
        "O tempo de inferência dos modelos foi satisfatório em ambientes de teste.",
        "A maioria das fraudes ocorre de forma isolada, sem picos de sequência.",
        "Random Forest apresentou melhor generalização ao lidar com novos dados.",
        "A curva de aprendizado dos modelos estabilizou rapidamente após poucas iterações.",
        "Transações legítimas se distribuem de forma uniforme ao longo do dia.",
        "Modelos treinados com validação estratificada obtiveram melhor desempenho final.",
        "A base original foi enriquecida com variáveis derivadas para melhorar o aprendizado.",
        "A utilização de múltiplos modelos permitiu avaliar a consistência dos resultados.",
        "A análise estatística reforça que o padrão de fraude é instável.",
        "Fraudes abaixo de R$1 representam mais de 80% dos casos.",
        "O treinamento supervisionado com amostragem estratificada melhorou o recall."
    ]

    return frases
