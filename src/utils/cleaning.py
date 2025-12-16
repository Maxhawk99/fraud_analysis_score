# src/utils/cleaning.py
# -*- coding: utf-8 -*-

"""
Utilitários de limpeza e padronização de dados para projetos de Análise/Ciência de Dados.

Principais recursos:
- Padronização de nomes de colunas (snake_case)
- Normalização de textos (strip/lower/sem acento)
- Cast de numéricos em pt-BR ("1.234,56" -> 1234.56)
- Parse de datas com timezone America/Belem
- Conversão robusta para boolean
- Padronização de categorias via dicionário de mapeamento
- Pipeline declarativo `basic_clean` para aplicar tudo de uma vez

Dependências:
- pandas
- unidecode (opcional; se não houver, um fallback simples é usado)
"""

from __future__ import annotations
import re
from typing import Dict, Iterable, Optional

import pandas as pd

try:
    from unidecode import unidecode
except Exception:
    # Fallback leve se unidecode não estiver instalado.
    def unidecode(x: str) -> str:  # type: ignore
        return x

__all__ = [
    "TZ",
    "show_help",
    "snake_case",
    "snake_case_columns",
    "normalize_text_series",
    "cast_numeric_ptbr",
    "cast_boolean",
    "parse_datetime_brt",
    "standardize_categories",
    "basic_clean",
]

TZ = "America/Belem"


def show_help(func) -> None:
    """
    Imprime o help (docstring) de uma função deste módulo.

    Exemplo
    -------
    >>> from src.utils import cleaning as C
    >>> C.show_help(C.cast_numeric_ptbr)
    (mostra o help no console)
    """
    print(func.__doc__ or "Sem docstring disponível.")


def snake_case(name: str) -> str:
    """
    Converte uma string para snake_case seguro (minúsculas, '_' entre palavras).

    Regras
    ------
    - Remove/padroniza caracteres especiais para "_"
    - Compacta múltiplos '_' consecutivos
    - Remove '_' no início/fim

    Parâmetros
    ----------
    name : str
        Texto original (ex.: "Data da Compra (BRL)").

    Retorna
    -------
    str
        Versão em snake_case (ex.: "data_da_compra_brl").

    Exemplo
    -------
    >>> snake_case(" Data da Compra (BRL) ")
    'data_da_compra_brl'
    """
    name = name.strip()
    name = re.sub(r"[^\w\s]", "_", name, flags=re.UNICODE)
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.lower().strip("_")


def snake_case_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna uma cópia do DataFrame com **todas as colunas** em snake_case.

    Parâmetros
    ----------
    df : pd.DataFrame

    Retorna
    -------
    pd.DataFrame

    Exemplo
    -------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"Data Compra": [1], "Valor (R$)": [10]})
    >>> snake_case_columns(df).columns.tolist()
    ['data_compra', 'valor_r']
    """
    out = df.copy()
    out.columns = [snake_case(c) for c in out.columns]
    return out


def normalize_text_series(
    s: pd.Series,
    lower: bool = True,
    strip: bool = True,
    remove_accents: bool = True,
) -> pd.Series:
    """
    Normaliza uma série textual: strip, lower e remoção de acentos.

    Parâmetros
    ----------
    s : pd.Series
        Série a normalizar (será convertida para dtype 'string').
    lower : bool, padrão=True
        Converte para minúsculas.
    strip : bool, padrão=True
        Remove espaços em branco nas extremidades.
    remove_accents : bool, padrão=True
        Remove acentuação (usa unidecode se disponível).

    Retorna
    -------
    pd.Series (dtype 'string')

    Exemplo
    -------
    >>> import pandas as pd
    >>> s = pd.Series(["  São Paulo  ", None, "MANAUS"])
    >>> normalize_text_series(s).tolist()
    ['sao paulo', <NA>, 'manaus']
    """
    x = s.astype("string")
    if strip:
        x = x.str.strip()
    if remove_accents:
        x = x.map(lambda v: None if pd.isna(v) else unidecode(v))
    if lower:
        x = x.str.lower()
    return x


def cast_numeric_ptbr(
    s: pd.Series,
    thousands: str = ".",
    decimal: str = ",",
) -> pd.Series:
    """
    Converte strings numéricas em formato **pt-BR** ("1.234,56") para float.

    Regras
    ------
    - Remove separador de milhar
    - Troca separador decimal por '.'
    - Converte com `pd.to_numeric(errors="coerce")` (valores inválidos viram NA)

    Parâmetros
    ----------
    s : pd.Series
    thousands : str, padrão="."
    decimal : str, padrão=","

    Retorna
    -------
    pd.Series (float)

    Exemplo
    -------
    >>> import pandas as pd
    >>> s = pd.Series(["1.234,56", "12,00", None, "abc"])
    >>> cast_numeric_ptbr(s).tolist()
    [1234.56, 12.0, nan, nan]
    """
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    x = s.astype("string").str.replace(thousands, "", regex=False).str.replace(decimal, ".", regex=False)
    return pd.to_numeric(x, errors="coerce")


def cast_boolean(
    s: pd.Series,
    truthy: Iterable[str] = ("true", "t", "1", "y", "yes", "sim"),
    falsy: Iterable[str] = ("false", "f", "0", "n", "no", "nao", "não"),
) -> pd.Series:
    """
    Converte textos/códigos comuns para boolean (dtype 'boolean' com NA).

    Parâmetros
    ----------
    s : pd.Series
    truthy : Iterable[str]
        Conjunto de valores interpretados como True (case-insensitive).
    falsy : Iterable[str]
        Conjunto de valores interpretados como False (case-insensitive).

    Retorna
    -------
    pd.Series (dtype 'boolean')

    Exemplo
    -------
    >>> import pandas as pd
    >>> s = pd.Series(["Sim", "nao", "YES", "0", "1", None, "talvez"])
    >>> cast_boolean(s).astype(object).tolist()
    [True, False, True, False, True, <NA>, <NA>]
    """
    if pd.api.types.is_bool_dtype(s):
        return s.astype("boolean")
    x = s.astype("string").str.strip().str.lower()
    truthy_set = set(map(str.lower, truthy))
    falsy_set = set(map(str.lower, falsy))
    def _map(v: Optional[str]):
        if v in truthy_set:
            return True
        if v in falsy_set:
            return False
        return pd.NA
    return x.map(_map).astype("boolean")


def parse_datetime_brt(
    s: pd.Series,
    dayfirst: bool = True,
    assume_localize_if_naive: bool = True,
    tz: str = TZ,
) -> pd.Series:
    """
    Converte strings/datetimes para pandas datetime com timezone **America/Belem**.

    Regras
    ------
    - Usa `pd.to_datetime(..., errors="coerce", dayfirst=dayfirst, infer_datetime_format=True)`
    - Se a série não tiver tz (naive) e `assume_localize_if_naive=True`, aplica `tz_localize(tz)`
      com `nonexistent="NaT"` e `ambiguous="NaT"`; se já tiver tz, usa `tz_convert(tz)`.
    - Valores inválidos viram `NaT`.

    Parâmetros
    ----------
    s : pd.Series
    dayfirst : bool, padrão=True
    assume_localize_if_naive : bool, padrão=True
    tz : str, padrão="America/Belem"

    Retorna
    -------
    pd.Series (datetime64[ns, America/Belem])

    Exemplo
    -------
    >>> import pandas as pd
    >>> s = pd.Series(["10/09/2025 22:15", "2025-09-01T08:00:00", "31/02/2025", None])
    >>> dt = parse_datetime_brt(s, dayfirst=True)
    >>> dt.dt.tz  # doctest: +ELLIPSIS
    <DstTzInfo 'America/Belem' ...>
    """
    dt = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst, infer_datetime_format=True, utc=False)
    try:
        current_tz = getattr(dt.dt, "tz", None)
        if current_tz is None and assume_localize_if_naive:
            dt = dt.dt.tz_localize(tz, nonexistent="NaT", ambiguous="NaT")
        elif current_tz is not None:
            dt = dt.dt.tz_convert(tz)
    except Exception:
        # Series toda NaT ou problemas de tz — devolve como está.
        pass
    return dt


def standardize_categories(
    s: pd.Series,
    mapping: Optional[Dict[str, str]] = None,
    to_upper: bool = False,
    to_lower: bool = True,
) -> pd.Series:
    """
    Padroniza categorias (strings) aplicando um dicionário de mapeamento e caixa.

    Parâmetros
    ----------
    s : pd.Series
    mapping : dict[str, str] | None
        Ex.: {"sao paulo": "sp", "sp": "sp"}
    to_upper : bool, padrão=False
        Força caixa alta após mapeamento (ignora to_lower).
    to_lower : bool, padrão=True
        Força caixa baixa após mapeamento.

    Retorna
    -------
    pd.Series (dtype 'string')

    Exemplo
    -------
    >>> import pandas as pd
    >>> s = pd.Series(["São Paulo", "sp", "Rio de Janeiro"])
    >>> mapping = {"sao paulo": "sp", "rio de janeiro": "rj"}
    >>> standardize_categories(normalize_text_series(s), mapping=mapping).tolist()
    ['sp', 'sp', 'rj']
    """
    x = s.astype("string")
    if mapping:
        x = x.map(mapping).fillna(x)
    if to_upper:
        x = x.str.upper()
    elif to_lower:
        x = x.str.lower()
    return x


def basic_clean(
    df: pd.DataFrame,
    *,
    datetime_cols: Iterable[str] = (),
    numeric_ptbr_cols: Iterable[str] = (),
    bool_cols: Iterable[str] = (),
    text_cols: Iterable[str] = (),
    category_mappings: Optional[Dict[str, Dict[str, str]]] = None,
    dayfirst: bool = True,
) -> pd.DataFrame:
    """
    Pipeline **declarativo** de limpeza básica:
    - Converte nomes de colunas para snake_case
    - Normaliza textos (strip/lower/sem acento)
    - Converte numéricos pt-BR para float
    - Faz parse de datas e ajusta timezone para America/Belem
    - Converte para boolean conforme listas truthy/falsy padrão
    - Aplica mapeamentos de categorias

    Parâmetros
    ----------
    df : pd.DataFrame
    datetime_cols : Iterable[str]
        Colunas a processar com `parse_datetime_brt`.
    numeric_ptbr_cols : Iterable[str]
        Colunas a processar com `cast_numeric_ptbr`.
    bool_cols : Iterable[str]
        Colunas a processar com `cast_boolean`.
    text_cols : Iterable[str]
        Colunas a normalizar via `normalize_text_series`.
    category_mappings : dict[str, dict[str, str]] | None
        Dicionário de mapeamentos por coluna (aplicado após texto/case).
    dayfirst : bool, padrão=True
        Usado no parse de datas.

    Retorna
    -------
    pd.DataFrame
        Novo DataFrame limpo/padronizado.

    Exemplo
    -------
    >>> import pandas as pd
    >>> raw = pd.DataFrame({
    ...   "Data Compra": ["10/09/2025 22:15", "31/02/2025", None],
    ...   "Valor (R$)": ["1.234,56", "12,00", "abc"],
    ...   "Estado": ["São Paulo", "sp", "Rio de Janeiro"],
    ...   "Aprovado?": ["Sim", "nao", None],
    ...   "Cidade": ["  Belém  ", "Manaus", None],
    ... })
    >>> clean = basic_clean(
    ...   raw,
    ...   datetime_cols=["Data Compra"],
    ...   numeric_ptbr_cols=["Valor (R$)"],
    ...   bool_cols=["Aprovado?"],
    ...   text_cols=["Cidade", "Estado"],
    ...   category_mappings={"estado": {"sao paulo":"sp", "rio de janeiro":"rj"}},
    ...   dayfirst=True,
    ... )
    >>> sorted(clean.columns)
    ['aprovado', 'cidade', 'data_compra', 'estado', 'valor_r']
    >>> clean["valor_r"].tolist()
    [1234.56, 12.0, nan]
    """
    category_mappings = category_mappings or {}
    out = snake_case_columns(df)

    # Textos primeiro (para alimentar mapeamentos em caixa/máscara estáveis)
    for c in text_cols:
        if c in out.columns:
            out[c] = normalize_text_series(out[c], lower=True, strip=True, remove_accents=True)

    # Numéricos pt-BR
    for c in numeric_ptbr_cols:
        if c in out.columns:
            out[c] = cast_numeric_ptbr(out[c])

    # Datas com tz America/Belem
    for c in datetime_cols:
        if c in out.columns:
            out[c] = parse_datetime_brt(out[c], dayfirst=dayfirst)

    # Booleanos
    for c in bool_cols:
        if c in out.columns:
            out[c] = cast_boolean(out[c])

    # Mapeamentos de categorias (após normalização de texto)
    for c, mapping in category_mappings.items():
        if c in out.columns:
            out[c] = standardize_categories(out[c], mapping=mapping, to_lower=True)

    return out
