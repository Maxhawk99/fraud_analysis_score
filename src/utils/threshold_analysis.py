import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def _confusion_counts(df, threshold, target_col, score_col):
    """
    Retorna VP, FP, FN, VN para um dado threshold.
    Regra: bloquear se df[score_col] >= threshold.
    """
    y_true = df[target_col].values
    y_pred = (df[score_col] >= threshold).astype(int)

    TP = np.sum((y_true == 1) & (y_pred == 1))  # Verdadeiro Positivo
    FP = np.sum((y_true == 0) & (y_pred == 1))  # Falso Positivo
    FN = np.sum((y_true == 1) & (y_pred == 0))  # Falso Negativo
    TN = np.sum((y_true == 0) & (y_pred == 0))  # Verdadeiro Negativo

    return TP, FP, FN, TN

def metrics_for_threshold(df, threshold, target_col, score_col):
    """
    Calcula métricas de classificação para um threshold do score.
    Regra: bloquear se df[score_col] >= threshold.

    Parâmetros:
    - df: DataFrame
    - threshold: valor de corte do score (int ou float)
    - target_col: nome da coluna alvo (0/1)
    - score_col: nome da coluna do score
    """

    preds = (df[score_col] >= threshold).astype(int)
    y_true = df[target_col]
    
    TN, FP, FN, TP = confusion_matrix(y_true, preds).ravel()
    
    total = TP + FP + FN + TN

    # Métricas principais
    sensibilidade = TP / (TP + FN) if (TP + FN) > 0 else 0           # Recall / TPR
    taxa_fp       = FP / (FP + TN) if (FP + TN) > 0 else 0           # FPR
    precisao      = TP / (TP + FP) if (TP + FP) > 0 else 0           # PPV
    f1            = (2 * precisao * sensibilidade / (precisao + sensibilidade)
                     if (precisao + sensibilidade) > 0 else 0)
    acuracia      = (TP + TN) / total if total > 0 else 0
    especificidade = TN / (TN + FP) if (TN + FP) > 0 else 0          # TNR

    trans_bloqueadas = TP + FP

    return {
        'threshold': threshold,
        'transacoes_bloqueadas': trans_bloqueadas,
        'fraudes_detectadas': TP,
        'acuracia': acuracia,
        'especificidade': especificidade,
        'taxa_fp': taxa_fp,
        'sensibilidade': sensibilidade,
        'precisao': precisao,
        'f1_score': f1   
    }


def analyze_thresholds(df, thresholds, target_col, score_col):
    """
    Calcula métricas para todos os thresholds desejados.

    Parâmetros:
    - df: DataFrame
    - thresholds: iterável de thresholds a testar
    - target_col: nome da coluna alvo (0/1)
    - score_col: nome da coluna do score

    Retorna:
    - DataFrame com métricas em percentual (exceto contagens)
    """

    resultados = [
        metrics_for_threshold(df, t, target_col, score_col)
        for t in thresholds
    ]

    threshold_df = pd.DataFrame(resultados)

    # Converter métricas para percentuais
    for col in ['sensibilidade', 'taxa_fp', 'precisao', 'f1_score',
                'acuracia', 'especificidade']:
        threshold_df[col] = (threshold_df[col] * 100).round(1)

    # Renomear colunas para *_pct
    threshold_df = threshold_df.rename(columns={
        'sensibilidade': 'sensibilidade_pct',
        'taxa_fp': 'taxa_fp_pct',
        'precisao': 'precisao_pct',
        'f1_score': 'f1_pct',
        'acuracia': 'acuracia_pct',
        'especificidade': 'especificidade_pct',
    })

    return threshold_df


def display_threshold_table(df):
    """
    Estiliza DataFrame de thresholds para visualização em notebook.
    """

    return df.style.format({
        'sensibilidade_pct': '{:.1f}%',
        'taxa_fp_pct': '{:.1f}%',
        'precisao_pct': '{:.1f}%',
        'f1_pct': '{:.1f}%',
        'acuracia_pct': '{:.1f}%',
        'especificidade_pct': '{:.1f}%',
    })

def confusion_table(df, thresholds, target_col, score_col):
    """
    Gera tabela com VP, FP, FN, VN para cada threshold.

    Retorna um DataFrame com colunas:
    ['threshold', 'VP', 'FP', 'FN', 'VN']
    """
    linhas = []
    for t in thresholds:
        TP, FP, FN, TN = _confusion_counts(df, t, target_col, score_col)
        linhas.append({
            'threshold': t,
            'TP': TP,
            'FP': FP,
            'FN': FN,
            'TN': TN,
        })

    return pd.DataFrame(linhas)

def economic_calibration(
    df,
    thresholds,
    target_col,
    score_col,
    saving_per_TP,   # R$ de fraude evitada por transação fraudulenta bloqueada
    cost_per_FP,     # R$ de custo / atrito por bloquear uma transação legítima
    cost_per_FN=None      # R$ de custo por fraude que passa; se None, assume igual ao TP_saving
):
    """
    Calcula impacto econômico para cada threshold.

    Retorna DataFrame com:
    ['threshold', 'VP', 'FP', 'FN', 'VN',
     'TP_saving', 'FP_cost', 'FN_cost', 'net_profit']
    """
    if cost_per_FN is None:
        cost_per_FN = saving_per_TP

    linhas = []
    for t in thresholds:
        TP, FP, FN, TN = _confusion_counts(df, t, target_col, score_col)

        TP_saving = TP * saving_per_TP
        FP_cost    = FP * cost_per_FP
        FN_cost    = FN * cost_per_FN
        net_profit   = TP_saving - FP_cost - FN_cost

        linhas.append({
            'threshold': t,
            'TP': TP,
            'FP': FP,
            'FN': FN,
            'TN': TN,
            'TP_saving': TP_saving,
            'FP_cost': FP_cost,
            'FN_cost': FN_cost,
            'net_profit': net_profit
        })

    econ_df = pd.DataFrame(linhas)
    return econ_df.sort_values('threshold')

def plot_confusion_matrix(df, threshold, score_col, target_col):
    """
    Plota matriz de confusão para o threshold escolhido.
    """
    preds = (df[score_col] >= threshold).astype(int)
    y_true = df[target_col]

    cm = confusion_matrix(y_true, preds)
    labels = ["0", "1"]

    plt.figure(figsize=(6,4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=labels,
        yticklabels=labels
    )

    plt.title(f"Confusion Matrix (threshold = {threshold})")
    plt.xlabel("Predicted Values")
    plt.ylabel("Real Values")
    plt.show()
