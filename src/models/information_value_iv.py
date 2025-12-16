import pandas as pd
import numpy as np


def interpret_iv(iv):
    """Retorna interpretação textual para o IV."""
    if iv < 0.02:
        return "< 0,02: Muito fraco (Baixo poder explicativo)"
    elif iv < 0.10:
        return "0,02–0.1: Fraco"
    elif iv < 0.30:
        return "0,1–0,3: Médio"
    elif iv < 0.50:
        return "0,3–0,5: Forte"
    else:
        return "> 0,5: Muito forte (requer atenção)"


def calc_iv_feature(series, target_series, bins=10):
    """Cálculo do IV para uma feature (chamada auxiliar)."""

    # Detecta variável binária
    if series.nunique() <= 2:
        binned = series
    else:
        try:
            binned = pd.qcut(series, q=bins, duplicates="drop")
        except:
            binned = pd.cut(series, bins=bins)

    df_temp = pd.DataFrame({"feature": binned, "target": target_series})

    grouped = df_temp.groupby("feature")["target"].agg(["count", "sum"])
    grouped = grouped.rename(columns={"count": "total", "sum": "event"})
    grouped["non_event"] = grouped["total"] - grouped["event"]

    # suavização
    grouped["event"] = grouped["event"].replace(0, 0.5)
    grouped["non_event"] = grouped["non_event"].replace(0, 0.5)

    total_event = grouped["event"].sum()
    total_non_event = grouped["non_event"].sum()

    grouped["dist_event"] = grouped["event"] / total_event
    grouped["dist_non_event"] = grouped["non_event"] / total_non_event

    grouped["woe"] = np.log(grouped["dist_event"] / grouped["dist_non_event"])
    grouped["iv_bin"] = (grouped["dist_event"] - grouped["dist_non_event"]) * grouped["woe"]

    iv_total = grouped["iv_bin"].sum()

    return iv_total, grouped


def calc_iv_all(df, target, bins=10):
    """Retorna DataFrame com IV + Interpretação automática."""
    results = []

    for col in df.columns:
        if col == target:
            continue

        try:
            iv_val, _ = calc_iv_feature(df[col], df[target], bins)
            results.append((col, iv_val, interpret_iv(iv_val)))
        except Exception:
            results.append((col, np.nan, "Erro"))

    iv_df = pd.DataFrame(results, columns=["feature", "iv", "definition"])
    return iv_df.sort_values("iv", ascending=False).reset_index(drop=True)
