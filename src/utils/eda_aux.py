import pandas as pd

def analyze_features(
    data,
    target: str,
    remove_features=None,
    discrete_cardinality: int = 10,
    # novo parâmetro, mais explícito:
    force_all_integers_discrete: bool = False,
    show: bool = True,
):
    remove_features = remove_features or []

    cols_all = data.columns.tolist()
    cols_removed = [c for c in remove_features if c in data.columns]
    cols_after_remove = [c for c in cols_all if c not in set(cols_removed)]

    num_cols = data.select_dtypes(include=['number']).columns.tolist()
    cat_cols = data.select_dtypes(include=['object', 'category', 'string', 'bool']).columns.tolist()
    dt_cols = [c for c in cols_after_remove if pd.api.types.is_datetime64_any_dtype(data[c])]
    td_cols = [c for c in cols_after_remove if pd.api.types.is_timedelta64_dtype(data[c])]

    known = set(num_cols) | set(cat_cols) | set(dt_cols) | set(td_cols)
    others_cols = [c for c in cols_after_remove if c not in known and c != target]

    def _is_discrete(col: str) -> bool:
        if col not in num_cols or col in cols_removed or col == target:
            return False
        uniq = data[col].nunique(dropna=True)

        # regra para inteiros
        if pd.api.types.is_integer_dtype(data[col]):
            if force_all_integers_discrete:
                return True
            # padrão: respeita o limiar
            return uniq <= discrete_cardinality

        # para floats, usa apenas o limiar
        return uniq <= discrete_cardinality

    discrete_num_cols = [c for c in num_cols if c in cols_after_remove and _is_discrete(c)]
    numeric_cols_final = [c for c in num_cols if c in cols_after_remove and c not in discrete_num_cols and c != target]
    categorical_cols_final = [c for c in cat_cols if c in cols_after_remove and c != target]
    datetime_cols_final = [c for c in dt_cols if c != target]
    timedelta_cols_final = [c for c in td_cols if c != target]

    summary = {
        "n_features_total": len(cols_all),
        "removed_features": cols_removed,
        "numeric_features": numeric_cols_final,
        "discrete_numeric_as_categorical": discrete_num_cols,
        "categorical_features": categorical_cols_final,
        "datetime_features": datetime_cols_final,
        "timedelta_features": timedelta_cols_final,
        "other_features": others_cols,
        "target": target if target in data.columns else None,
        "counts": {
            "removed": len(cols_removed),
            "numeric": len(numeric_cols_final),
            "discrete": len(discrete_num_cols),
            "categorical": len(categorical_cols_final),
            "datetime": len(datetime_cols_final),
            "timedelta": len(timedelta_cols_final),
            "others": len(others_cols),
        },
    }

    if show:
        print(f"There are {summary['n_features_total']} features.\n")
        print(f"{summary['counts']['numeric']} numeric feature(s)."); print(summary["numeric_features"], "\n")
        print(f"{summary['counts']['discrete']} numeric feature(s) considered categorical (discrete)."); print(summary["discrete_numeric_as_categorical"], "\n")
        print(f"{summary['counts']['categorical']} categorical feature(s)."); print(summary["categorical_features"], "\n")
        print(f"{summary['counts']['datetime']} datetime feature(s)."); print(summary["datetime_features"], "\n")
        print(f"{summary['counts']['timedelta']} timedelta feature(s)."); print(summary["timedelta_features"], "\n")
        print(f"{summary['counts']['others']} feature(s) that don't fit the above."); print(summary["other_features"], "\n")
        print(f"{summary['counts']['removed']} removed feature(s)."); print(summary["removed_features"], "\n")
        print("Target feature:"); print(summary["target"])

    return summary
