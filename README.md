# 🚨 Score Manual de Fraude com Validação Estatística e Econômica

## 1. Visão Geral

Este projeto tem como objetivo desenvolver um score manual de risco de fraude para transações financeiras, combinando análise exploratória, engenharia de features, métricas estatísticas e calibração econômica.

Diferentemente de uma abordagem puramente baseada em Machine Learning, o foco aqui é construir um modelo interpretável, auditável e operacionalmente viável, adequado para cenários com dataset reduzido, realidade comum em projetos iniciais de fraude.

## 2. Problema de Negócio

Fraudes financeiras geram prejuízos diretos e impacto negativo na experiência do cliente.
O desafio é identificar transações fraudulentas com eficiência, equilibrando:

* Detecção de fraude
* Redução de falsos positivos
* Viabilidade operacional
* Impacto financeiro real

O objetivo é apoiar decisões como liberar, revisar ou bloquear transações, com base em níveis de risco.

## 3. Abordagem Utilizada

O projeto segue uma abordagem estruturada:

1. **Análise Exploratória (EDA)**
   Identificação de padrões comportamentais, temporais e financeiros associados a fraudes.

2. **Engenharia de Features**
   Criação de variáveis derivadas interpretáveis (ex.: comportamento do usuário, recorrência, anomalias de valor).

3. **Score Manual de Risco**
   Construção de um sistema de pontuação baseado em:

   * Risco Relativo (RR)
   * Information Value (IV)
   * Matriz de Correlação
     Pesos discretos (1, 2, 3) garantem simplicidade e interpretabilidade.

4. **Validação Estatística**
   Comparação do score manual com um modelo de Regressão Logística, avaliando convergência de importância e performance.

5. **Calibração Econômica**
   Otimização do threshold com base em impacto financeiro, e não apenas métricas estatísticas.

## 4. Principais Resultados

* Forte convergência entre score manual e modelo estatístico

![curvas_roc](figures/curvas_roc_score_manual_x_regressao_logistica.png)

* **AUC – Regressão Logística:** ~0.90
* **AUC – Score Manual:** ~0.90
* **R$ 822.165** em redução total de perdas financeiras em relação ao cenário sem score.
* Threshold definido com base em **lucro líquido máximo**, e não apenas F1-score.

O score mostrou-se financeiramente viável, com potencial de se pagar já no primeiro período de operação, dependendo do custo de implementação.

## 5. Estrutura dos Notebooks do Projeto

```
├── notebooks/
├── 00_intro_entendimento_dados.ipynb
├── 01_eda.ipynb
├── 02_engenharia_features.ipynb
└── 03_score_avaliacao_performance.ipynb
```

## 6. Tecnologias Utilizadas

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib / Seaborn
* Jupyter Notebook

## 7. Conclusão

Este projeto demonstra que, mesmo com dados limitados, é possível construir um sistema de decisão robusto ao combinar estatística, interpretação de negócio e avaliação econômica.

A abordagem manual, quando bem fundamentada, pode ser tão eficaz quanto modelos de Machine Learning, oferecendo maior transparência, controle e facilidade de implementação em ambientes reais de fraude.

## 8. Próximos Passos

* Testar o score em dados temporais futuros
* Automatizar o pipeline de scoring
* Avaliar integração com modelos supervisionados em produção

## 📫 9. Contato

- [LinkedIn](https://www.linkedin.com/in/marx-araujo/)
- [GitHub](https://github.com/Maxhawk99)
- [Portfolio](https://merciful-daphne-98e.notion.site/Marx-Araujo-225e883ebb298090b128c34d2eb3b864)
- [Medium](https://medium.com/@marx.araujo99)
- [Email](mailto:marx.araujo99@gmail.com)

