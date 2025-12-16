# src/utils/profiling.py
# -*- coding: utf-8 -*-

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Optional
import warnings

__all__ = [
    "_mixed_types",
    "_can_parse_numeric_ptbr",
    "_can_parse_datetime",
    "profile_df",
    "profile_parse",
    "list_functions",
]

def list_functions() -> None:
    """
    Exibe todas as funções públicas disponíveis no módulo com um breve resumo.
    
    Exemplo
    -------
    >>> from src.utils import profiling
    >>> profiling.list_functions()
    """
    functions_info = {
        "profile_df": {
            "resumo": "Gera relatório completo de perfilamento com tipos, nulos, cardinalidade, etc.",
            "uso": "profile_df(df)",
            "retorna": "pd.DataFrame"
        },
        "profile_parse": {
            "resumo": "Analisa parseabilidade de colunas object para números pt-BR e/ou datas.",
            "uso": "profile_parse(df, cols=None, check_numeric=True, check_datetime=True)",
            "retorna": "pd.DataFrame"
        },
    }
    
    print("\n" + "=" * 80)
    print("FUNÇÕES DISPONÍVEIS EM profiling.py")
    print("=" * 80)
    
    for i, (func_name, info) in enumerate(functions_info.items(), 1):
        print(f"\n{i}. {func_name}()")
        print(f"   📝 {info['resumo']}")
        print(f"   💻 Uso: {info['uso']}")
        print(f"   ↩️ Retorna: {info['retorna']}")
    
    print("\n" + "=" * 80)
    print("Para mais detalhes, use: help(profile_df) ou help(profile_parse)")
    print("=" * 80 + "\n")


def _mixed_types(s: pd.Series, sample: int = 500) -> bool:
    """
    Detecta se uma Série do pandas (coluna) possui **tipos Python mistos** 
    quando seu dtype é 'object' (ex.: [int, float, str] misturados).

    Parâmetros
    ----------
    s : pd.Series
        Coluna a ser inspecionada.
    sample : int, padrão=500
        Máximo de amostras não nulas para inspecionar tipos subjacentes.

    Retorna
    -------
    bool
        True se houver pelo menos dois tipos Python distintos (ex.: 'int' e 'str').
        False caso contrário (inclui casos em que todos os valores são strings).

    Observações
    -----------
    - Esta função verifica mistura de **tipos Python**, não "mistura semântica".
      Se a coluna tem apenas strings (mesmo que representem números/datas), 
      o resultado será False.
    - Para "mistura semântica" (parte parseia, parte não), veja as funções
      `has_mixed_numeric_semantics` e `has_mixed_datetime_semantics`.

    Exemplo
    -------
    >>> import pandas as pd
    >>> s = pd.Series([1, "2", 3.0, None, "texto"])
    >>> _mixed_types(s)
    True
    >>> s2 = pd.Series(["1", "2", "3"])
    >>> _mixed_types(s2)
    False
    """
    if s.dtype != "object":
        return False
    non_null = s.dropna()
    if non_null.empty:
        return False
    types = non_null.sample(min(sample, non_null.shape[0]), random_state=0) \
                    .map(lambda x: type(x).__name__).value_counts()
    return types.shape[0] > 1


def _can_parse_numeric_ptbr(
    s: pd.Series, sample: int = 1000, sep_th: str = ".", dec_sep: str = ","
) -> float:
    """
    Estima a fração (0–1) de valores que podem ser convertidos de **formato pt-BR** 
    (ex.: "1.234,56") para número (float).

    Parâmetros
    ----------
    s : pd.Series
        Coluna com possíveis números em formato pt-BR.
    sample : int, padrão=1000
        Nº máximo de amostras não nulas para teste de parse.
    sep_th : str, padrão='.'
        Separador de milhar esperado nas strings.
    dec_sep : str, padrão=','
        Separador decimal esperado nas strings.

    Retorna
    -------
    float
        Proporção entre 0.0 e 1.0 de valores parseáveis; `np.nan` se não houver amostras.

    Observações
    -----------
    - Se `s` já for numérica (dtype float/int), retorna 1.0.
    - Útil para decidir se o cast pt-BR → float é seguro na etapa de limpeza.

    Exemplo
    -------
    >>> import pandas as pd
    >>> s = pd.Series(["1.234,56", "12,00", None, "abc", "2.500,0"])
    >>> round(_can_parse_numeric_ptbr(s), 2)
    0.75
    """
    if s.dtype in ("float64", "float32", "int64", "int32"):
        return 1.0
    x = s.dropna().astype(str).head(sample)
    if x.empty:
        return np.nan
    x = x.str.replace(sep_th, "", regex=False).str.replace(dec_sep, ".", regex=False)
    parsed = pd.to_numeric(x, errors="coerce")
    return float(parsed.notna().mean())


def _can_parse_datetime(s, sample=1000, dayfirst=True, fmt=None):
    """
    Estima a fração (0–1) de valores que podem ser convertidos para **datetime** 
    com `pandas.to_datetime`, respeitando `dayfirst`.

    Parâmetros
    ----------
    s : pd.Series
        Coluna com possíveis datas em string.
    sample : int, padrão=1000
        Nº máximo de amostras não nulas para teste de parse.
    dayfirst : bool, padrão=True
        Se True, interpreta "10/09/2025" como 10 de setembro (pt-BR).

    Retorna
    -------
    float
        Proporção entre 0.0 e 1.0 de valores parseáveis; `np.nan` se não houver amostras.

    Exemplo
    -------
    >>> import pandas as pd
    >>> s = pd.Series(["10/09/2025", "31/02/2025", "2025-09-01", None, "09-10-2025"])
    >>> round(_can_parse_datetime(s, dayfirst=True), 2)
    0.25
    """
    x = s.dropna().astype(str).head(sample)
    if x.empty:
        return np.nan
    parsed = pd.to_datetime(
        x,
        errors="coerce",
        dayfirst=dayfirst,
        format=fmt  # ex: "%d/%m/%Y"
    )
    return float(parsed.notna().mean())


def profile_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gera um **relatório de perfilamento** por coluna com:
    - tipo (`dtype`)
    - nº/percentual de nulos
    - nº de valores únicos e razão de cardinalidade
    - amostra de valores
    - diagnóstico de **tipos mistos** (estrutural)
    - `min`/`max` quando aplicável

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame a ser perfilado.

    Retorna
    -------
    pd.DataFrame
        Relatório por coluna.

    Exemplo
    -------
    >>> import pandas as pd
    >>> demo = pd.DataFrame({
    ...     "id": [1, 1, 2, 3, 4],
    ...     "a_mixed": [1, "2", 3.0, None, "texto"],
    ...     "b_num_ptbr": ["1.234,56", "12,00", None, "abc", "2.500,0"],
    ...     "d_numeric": [10, 20, 30, 40, 50],
    ... })
    >>> report = profile_df(demo)
    >>> report.loc[report["col"].eq("d_numeric"), "dtype"].iloc[0]
    'int64'
    """
    rows = []
    n_rows = len(df)

    for col in df.columns:
        s = df[col]
        n_null = int(s.isna().sum())
        non_null_count = int(s.notna().sum())
        n_unique_non_null = int(s.dropna().nunique())
        dup_rate_col = (
            (non_null_count - n_unique_non_null) / non_null_count
            if non_null_count > 0 else 0.0
        )

        row = {
            "col": col,
            "dtype": str(s.dtype),
            "n_rows": n_rows,
            "n_null": n_null,
            "pct_null": round(n_null / n_rows, 4) if n_rows else 0.0,
            "n_unique": n_unique_non_null,
            "cardinality_ratio": round(n_unique_non_null / n_rows, 4) if n_rows else 0.0,
            "dup_rate_col": round(dup_rate_col, 4),
            "sample_values": s.dropna().astype(str).head(5).tolist(),
            "has_mixed_types": _mixed_types(s),
            "min": None,
            "max": None,
        }

        try:
            if pd.api.types.is_numeric_dtype(s):
                row["min"], row["max"] = float(s.min()), float(s.max())
            elif pd.api.types.is_datetime64_any_dtype(s):
                row["min"], row["max"] = s.min(), s.max()
        except Exception:
            pass

        rows.append(row)

    prof = pd.DataFrame(rows)
    return prof


def profile_parse(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    check_numeric: bool = True,
    check_datetime: bool = True,
    dayfirst: bool = True,
) -> pd.DataFrame:
    """
    Analisa a **parseabilidade** de colunas tipo 'object' para números pt-BR e/ou datas.
    
    Por padrão, seleciona automaticamente apenas colunas com dtype='object'.
    Útil para identificar colunas "disfarçadas" que podem ser convertidas.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame a ser analisado.
    cols : list[str] | None, padrão=None
        Lista de colunas específicas para analisar. Se None, usa todas as colunas 'object'.
    check_numeric : bool, padrão=True
        Se True, inclui coluna 'can_parse_num_ptbr' no relatório.
    check_datetime : bool, padrão=True
        Se True, inclui coluna 'can_parse_datetime' no relatório.
    dayfirst : bool, padrão=True
        Usado nos testes de datetime (interpreta "10/09/2025" como 10 de setembro).

    Retorna
    -------
    pd.DataFrame
        Relatório com colunas: 'col', 'dtype', e as colunas de parse solicitadas.
        Retorna vazio se não houver colunas 'object' elegíveis.

    Exemplo
    -------
    >>> import pandas as pd
    >>> demo = pd.DataFrame({
    ...     "a_mixed": [1, "2", 3.0, None, "texto"],
    ...     "b_num_ptbr": ["1.234,56", "12,00", None, "abc", "2.500,0"],
    ...     "c_dates": ["10/09/2025", "31/02/2025", "2025-09-01", None, "09-10-2025"],
    ...     "d_numeric": [10, 20, 30, 40, 50],
    ... })
    >>> report = profile_parse(demo)
    >>> report.shape[0]  # deve ter 3 linhas (apenas colunas object)
    3
    >>> report = profile_parse(demo, cols=["b_num_ptbr"], check_datetime=False)
    >>> "can_parse_datetime" in report.columns
    False
    """
    # Selecionar colunas
    if cols is None:
        cols = [c for c in df.columns if df[c].dtype == "object"]
    else:
        # Validar que as colunas existem
        missing = set(cols) - set(df.columns)
        if missing:
            raise ValueError(f"Colunas não encontradas no DataFrame: {missing}")
    
    if not cols:
        warnings.warn("Nenhuma coluna 'object' encontrada para análise de parse.")
        return pd.DataFrame(columns=["col", "dtype"])
    
    rows = []
    for col in cols:
        s = df[col]
        row = {
            "col": col,
            "dtype": str(s.dtype),
        }
        
        if check_numeric:
            row["can_parse_num_ptbr"] = round(_can_parse_numeric_ptbr(s), 3)
        
        if check_datetime:
            row["can_parse_datetime"] = round(_can_parse_datetime(s, dayfirst=dayfirst), 3)
        
        rows.append(row)
    
    return pd.DataFrame(rows)