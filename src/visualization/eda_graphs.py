'''
This script aims to provide functions that will turn the exploratory data analysis (EDA) process easier. 
'''


'''
Importing libraries
'''

# Data manipulation and visualization.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from IPython.display import display, HTML
import math
from matplotlib.ticker import FuncFormatter
from matplotlib.font_manager import FontProperties
from typing import Optional, Tuple, List, Dict

# Debugging.
from .exception import CustomException

# Warnings.
from warnings import filterwarnings
filterwarnings('ignore')

def _fmt_num(x, decimal_places=2):
    """
    Normaliza valores muito próximos de zero para 0
    e remove sinais negativos indesejados em rótulos.
    """
    try:
        if abs(float(x)) < 10**(-decimal_places) + 1e-12:
            x = 0.0
        s = str(round(float(x), decimal_places))
        s = s.replace('-0.0', '0.0').replace('-0', '0')
        return s
    except Exception:
        return str(x)

palette=sns.color_palette(['#27457B', '#e85d04', '#0077b6', '#ff8200', '#0096c7', '#ff9c33'])

def analysis_plots(data, features, histplot=True, barplot=False, mean=None, text_y=0.5,    
                   outliers=False, boxplot=False, boxplot_x=None, kde=False, hue=None, 
                   nominal=False, color='#023047', figsize=None):
    '''
    Generate plots for univariate and bivariate analysis.

    This function generates histograms, horizontal bar plots 
    and boxplots based on the provided data and features. 

    Args:
        data (DataFrame): The DataFrame containing the data to be visualized.
        features (list): A list of feature names to visualize.
        histplot (bool, optional): Generate histograms. Default is True.
        barplot (bool, optional): Generate horizontal bar plots. Default is False.
        mean (bool, optional): Generate mean bar plots of specified feature instead of proportion bar plots. Default is None.
        text_y (float, optional): Y coordinate for text on bar plots. Default is 0.5.
        outliers (bool, optional): Generate boxplots for outliers visualization. Default is False.
        boxplot (bool, optional): Generate boxplots for categories distributions comparison. Default is False.
        boxplot_x (str, optional): The feature to which the categories will have their distributions compared. Default is None.
        kde (bool, optional): Plot Kernel Density Estimate in histograms. Default is False.
        hue (str, optional): Hue for histogram and bar plots. Default is None.
        color (str, optional): The color of the plot. Default is '#023047'.
        figsize (tuple, optional): The figsize of the plot. If None, auto-calculated based on columns.

    Returns:
        None

    Raises:
        CustomException: If an error occurs during the plot generation.

    '''
    
    try:
        # ------------------------------
        # Grid e figsize proporcional
        # ------------------------------
        num_features = len(features)
        cols = min(3, max(1, num_features))
        rows = math.ceil(num_features / cols)
        
        # Define figsize baseado no número de colunas e tipo de plot se não fornecido
        if figsize is None:
            if barplot:
                # Barplot: tamanhos específicos
                if cols == 1:
                    figsize = (4.27, 1.5)
                elif cols == 2:
                    figsize = (8.53, 1.5)
                else:  # 3 colunas
                    figsize = (10, 1.5)
            elif outliers:
                # Outliers: tamanhos específicos
                if cols == 1:
                    figsize = (3.4, 1.5)
                elif cols == 2:
                    figsize = (6.8, 1.5)
                else:  # 3 colunas
                    figsize = (10, 1.5)
            else:
                # Default (histplot): tamanhos específicos
                if cols == 1:
                    figsize = (3.4, 3.2)
                elif cols == 2:
                    figsize = (6.8, 3.2)
                else:  # 3 colunas
                    figsize = (10, 3)
        
        # Calcula figsize final baseado no número de linhas
        width, height = figsize
        actual_figsize = (width, height * rows)
        
        fig, axes = plt.subplots(rows, cols, figsize=actual_figsize)
        axes = np.atleast_2d(axes)
        
        # Tamanhos de fonte padronizados
        TITLE_SIZE = 8
        LABEL_SIZE = 7
        TICK_SIZE = 6
        VALUE_SIZE = 6
        LEGEND_SIZE = 6  

        for i, feature in enumerate(features):
            row = i // cols
            col = i % cols

            ax = axes[row, col] 
            
            if barplot:
                if mean:
                    data_grouped = data.groupby([feature])[[mean]].mean().reset_index()
                    data_grouped[mean] = round(data_grouped[mean], 2)
                    ax.barh(y=data_grouped[feature], width=data_grouped[mean], color=color)
                    for index, value in enumerate(data_grouped[mean]):
                        # Adjust the text position based on the width of the bars
                        ax.text(value + text_y, index, f'{value:.1f}', va='center', fontsize=VALUE_SIZE)
                else:
                    if hue:
                        data_grouped = data.groupby([feature])[[hue]].mean().reset_index().rename(columns={hue: 'pct'})
                        data_grouped['pct'] *= 100
                    else:
                        data_grouped = data.groupby([feature])[[feature]].count().rename(columns={feature: 'count'}).reset_index()
                        data_grouped['pct'] = data_grouped['count'] / data_grouped['count'].sum() * 100
        
                    ax.barh(y=data_grouped[feature], width=data_grouped['pct'], color=color)
                    
                    if pd.api.types.is_numeric_dtype(data_grouped[feature]):
                        ax.invert_yaxis()
                        
                    for index, value in enumerate(data_grouped['pct']):
                        # Adjust the text position based on the width of the bars
                        ax.text(value + text_y, index, f'{value:.1f}%', va='center', fontsize=VALUE_SIZE)
                
                ax.set_yticks(ticks=range(data_grouped[feature].nunique()), labels=data_grouped[feature].tolist(), fontsize=TICK_SIZE)
                ax.get_xaxis().set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.grid(False)
        
            elif outliers:
                # Plot univariate boxplot.
                sns.boxplot(data=data, x=feature, ax=ax, color=color)
                ax.tick_params(axis='both', labelsize=TICK_SIZE)
            
            elif boxplot:
                # Plot multivariate boxplot.
                sns.boxplot(data=data, x=boxplot_x, y=feature, showfliers=outliers, ax=ax, palette=palette)
                ax.tick_params(axis='both', labelsize=TICK_SIZE)

            else:
                # Plot histplot.
                sns.histplot(data=data, x=feature, kde=kde, ax=ax, color=color, stat='proportion', hue=hue, line_kws={'linewidth': 1})
                ax.tick_params(axis='both', labelsize=TICK_SIZE)
                
                # Configura legenda se houver hue
                if hue is not None:
                    # Pega a legenda existente para extrair handles e labels
                    old_legend = ax.get_legend()
                    if old_legend:
                        handles = old_legend.legend_handles
                        labels = [t.get_text() for t in old_legend.get_texts()]
                        title = old_legend.get_title().get_text()
                        
                        # Recria legenda compacta (igual binary_analysis)
                        legend = ax.legend(handles=handles, labels=labels, title=title,
                                         fontsize=LEGEND_SIZE, title_fontsize=LEGEND_SIZE,
                                         frameon=True, fancybox=False, shadow=False,
                                         edgecolor='gray', framealpha=0.9)
                        legend.get_frame().set_linewidth(0.5)

            ax.set_title(feature, fontsize=TITLE_SIZE)  
            ax.set_xlabel('', fontsize=LABEL_SIZE)
            ax.set_ylabel('', fontsize=LABEL_SIZE)
        
        # Remove unused axes.
        total_slots = rows * cols
        if num_features < total_slots:
            flat_axes = axes.flatten()
            for j in range(num_features, total_slots):
                fig.delaxes(flat_axes[j])

        plt.tight_layout()
    
    except Exception as e:
        raise CustomException(e, sys)

def check_outliers(data, features):
    '''
    Check for outliers in the given dataset features.

    This function calculates and identifies outliers in the specified features
    using the Interquartile Range (IQR) method.

    Args:
        data (DataFrame): The DataFrame containing the data to check for outliers.
        features (list): A list of feature names to check for outliers.

    Returns:
        tuple: A tuple containing three elements:
            - outlier_indexes (dict): A dictionary mapping feature names to lists of outlier indexes.
            - outlier_counts (dict): A dictionary mapping feature names to the count of outliers.
            - total_outliers (int): The total count of outliers in the dataset.

    Raises:
        CustomException: If an error occurs while checking for outliers.

    '''
    
    try:
    
        outlier_counts = {}
        outlier_indexes = {}
        total_outliers = 0
        
        for feature in features:
            Q1 = data[feature].quantile(0.25)
            Q3 = data[feature].quantile(0.75)
            
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            feature_outliers = data[(data[feature] < lower_bound) | (data[feature] > upper_bound)]
            outlier_indexes[feature] = feature_outliers.index.tolist()
            outlier_count = len(feature_outliers)
            outlier_counts[feature] = outlier_count
            total_outliers += outlier_count
        
        print(f'There are {total_outliers} outliers in the dataset.')
        print()
        print(f'Number (percentage) of outliers per feature: ')
        print()
        for feature, count in outlier_counts.items():
            print(f'{feature}: {count} ({round(count/len(data)*100, 2)})%')

        return outlier_indexes, outlier_counts, total_outliers
    
    except Exception as e:
        raise CustomException(e, sys)

def create_category_order(feature_name, categories_list):
    """
    Cria um dicionário de ordenação de categorias para usar em binary_analysis.
    
    Args:
        feature_name (str): Nome da feature/coluna.
        categories_list (list): Lista com as categorias na ordem desejada.
    
    Returns:
        dict: Dicionário {feature_name: categories_list} pronto para usar.
    
    Exemplo:
        >>> order = create_category_order('education', ['High School', 'Bachelor', 'Master', 'PhD'])
        >>> binary_analysis(data, 'education', 'target', barplot=True, category_order=order)
    """
    return {feature_name: categories_list}


def num_binary_analysis(
    data,
    features,
    hue,
    histplot=True,
    boxplot=True,
    table=True,
    stats_table=True,
    bins='auto',
    decimal_places=2,
    color='#27457B',
    figsize=None,
    xlim=None,
    wrap_cycle=None,
    step=None,
    start=None,
    head_cutoff=None,
    tail_cutoff=None,
    max_edge=None,
    table_name=None,
    stats_name=None
):
    """
    Análise bivariada para variáveis NUMÉRICAS com bins.
    
    Args:
        data (DataFrame): DataFrame com os dados.
        features (list|str): Lista de features numéricas para análise.
        hue (str): Nome da coluna target binária (0/1).
        histplot (bool): Exibir histograma com hue. Default True.
        boxplot (bool): Exibir boxplot com hue. Default True.
        table (bool): Exibir tabela de análise de risco. Default True.
        stats_table (bool): Exibir tabela de estatísticas descritivas por target. Default True.
        bins (int|str): Configuração de bins para tabela.
        decimal_places (int): Casas decimais.
        color (str): Cor base dos gráficos.
        figsize (tuple): Tamanho da figura.
        xlim (tuple): Limites do eixo X.
        wrap_cycle (int): Ciclo para rótulos circulares.
        step (float): Largura fixa dos bins.
        start (float): Valor inicial dos bins.
        head_cutoff (float): Cutoff inicial para bin "< valor".
        tail_cutoff (float): Cutoff final para bin "> valor".
        max_edge (float): Força última borda dos bins.
        table_name (str): Nome customizado para a tabela de risco.
        stats_name (str): Nome customizado para a tabela de estatísticas.
    
    Returns:
        dict: Dicionário com 'risk_tables' e 'stats_tables' (cada um pode ser DataFrame ou dict de DataFrames).
              Para acessar: result['risk_tables'] e result['stats_tables']
    """
    
    try:
        if isinstance(features, str):
            features = [features]
        
        risk_tables = {}
        stats_tables = {}
        
        for feature in features:
            # Verificar se é numérica
            if not pd.api.types.is_numeric_dtype(data[feature]):
                print(f"Warning: '{feature}' não é numérica. Use categorical_binary_analysis().")
                continue
            
            show_header = histplot or boxplot
            if show_header:
                print(f"\n{'─'*80}")
                print(f"Análise Bivariada Numérica: {feature}")
                print(f"{'─'*80}\n")
            
            # ===== GRÁFICOS =====
            if histplot or boxplot:
                n_plots = int(histplot) + int(boxplot)
                
                if figsize is None:
                    if n_plots == 2:
                        actual_figsize = (8.53, 3.2)
                    else:
                        actual_figsize = (4.27, 3.2)
                else:
                    actual_figsize = figsize
                
                if n_plots > 0:
                    fig, axes = plt.subplots(1, n_plots, figsize=actual_figsize)
                    
                    TITLE_SIZE = 8
                    LABEL_SIZE = 7
                    TICK_SIZE = 6
                    
                    if n_plots == 1:
                        axes = [axes]
                    
                    plot_idx = 0
                    
                    # HISTOGRAMA COM BINS
                    if histplot:
                        ax = axes[plot_idx]
                        
                        series = data[feature].dropna().to_numpy()
                        
                        # Construir edges
                        custom_edges = None
                        if step is not None:
                            smin = series.min() if start is None else start
                            smax = series.max()

                            bin_start = head_cutoff if head_cutoff is not None else smin
                            
                            if tail_cutoff is not None:
                                bin_end = tail_cutoff
                            elif max_edge is not None:
                                bin_end = max_edge
                            else:
                                bin_end = smax

                            import math
                            finite_edges = [float(bin_start)]
                            if step <= 0:
                                step = 1.0

                            current = float(bin_start)
                            while current < float(bin_end):
                                current += float(step)
                                finite_edges.append(current)

                            left = (-np.inf,) if head_cutoff is not None else ()
                            right = (np.inf,) if tail_cutoff is not None else ()
                            custom_edges = np.array(left + tuple(finite_edges) + right, dtype=float)
                            
                        elif (head_cutoff is not None) or (tail_cutoff is not None):
                            _, base_edges = np.histogram(series, bins=10 if bins == 'auto' else bins)
                            left = (-np.inf,) if head_cutoff is not None else ()
                            right = (np.inf,) if tail_cutoff is not None else ()
                            custom_edges = np.array(left + tuple(base_edges) + right, dtype=float)
                        
                        if custom_edges is not None:
                            edges = custom_edges
                        else:
                            if bins == 'auto':
                                n_bins = min(20, int(np.ceil(np.log2(len(series)) + 1)))
                                _, edges = np.histogram(series, bins=n_bins)
                            else:
                                _, edges = np.histogram(series, bins=bins)
                        
                        # Criar labels
                        bin_labels = []
                        for i in range(len(edges) - 1):
                            lo, hi = edges[i], edges[i + 1]
                            if np.isneginf(lo):
                                label = f"<{round(hi, decimal_places)}"
                            elif np.isposinf(hi):
                                label = f">{round(lo, decimal_places)}"
                            else:
                                if wrap_cycle is not None and float(int(lo)) == lo and float(int(hi)) == hi:
                                    lo_i = int(lo) % wrap_cycle
                                    hi_i = int(hi) % wrap_cycle
                                    label = f"{lo_i}-{hi_i}"
                                else:
                                    lo_str = _fmt_num(lo, decimal_places)
                                    hi_str = _fmt_num(hi, decimal_places)
                                    label = f"{lo_str}-{hi_str}"
                            bin_labels.append(label)
                        
                        right_flag = False if wrap_cycle is not None else True
                        
                        data_copy = data.copy()
                        data_copy['__bin'] = pd.cut(data_copy[feature], bins=edges, include_lowest=True, right=right_flag)
                        label_map = dict(zip(data_copy['__bin'].cat.categories, bin_labels))
                        data_copy['__bin_label'] = data_copy['__bin'].map(label_map)
                        
                        # Plotar
                        for hue_val in sorted(data[hue].unique()):
                            data_filtered = data_copy[data_copy[hue] == hue_val]
                            counts = data_filtered.groupby('__bin_label', observed=True).size()
                            counts = counts.reindex(bin_labels, fill_value=0)
                            proportions = counts / len(data) if len(data) > 0 else counts
                            
                            x_pos = np.arange(len(bin_labels))
                            ax.bar(x_pos, proportions, alpha=0.5, 
                                  label=f'{hue_val}',
                                  color=palette[int(hue_val)])
                        
                        num_bins = len(bin_labels)
                        if num_bins <= 10:
                            ax.set_xticks(np.arange(num_bins))
                            ax.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=TICK_SIZE)
                        else:
                            step_size = max(1, num_bins // 10)
                            tick_positions = np.arange(0, num_bins, step_size)
                            ax.set_xticks(tick_positions)
                            ax.set_xticklabels([bin_labels[i] for i in tick_positions], 
                                              rotation=45, ha='right', fontsize=TICK_SIZE)
                        
                        if xlim is not None:
                            ax.set_xlim(*xlim)
                        
                        ax.grid(axis='x', visible=False)
                        ax.grid(axis='y', visible=True, alpha=0.3)
                        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))
                        ax.tick_params(axis='y', labelsize=TICK_SIZE)
                        ax.set_ylabel("Proportion", fontsize=LABEL_SIZE)
                        ax.set_xlabel("", fontsize=LABEL_SIZE)
                        ax.set_title(feature, fontsize=TITLE_SIZE)
                        
                        legend = ax.legend(title=hue, fontsize=TICK_SIZE, title_fontsize=TICK_SIZE,
                                         frameon=True, fancybox=False, shadow=False,
                                         edgecolor='gray', framealpha=0.9)
                        legend.get_frame().set_linewidth(0.5)
                        
                        plot_idx += 1
                    
                    # BOXPLOT
                    if boxplot:
                        ax = axes[plot_idx]
                        sns.boxplot(data=data, x=hue, y=feature, ax=ax, palette=palette[:2])
                        ax.set_xlabel(hue, fontsize=LABEL_SIZE)
                        ax.set_ylabel(feature, fontsize=LABEL_SIZE)
                        ax.set_title(f'Boxplot: {feature} por {hue}', fontsize=TITLE_SIZE)
                        ax.tick_params(labelsize=TICK_SIZE)
                    
                    plt.tight_layout()
                    plt.show()
            
            # ===== TABELA DE ESTATÍSTICAS DESCRITIVAS (ao lado da tabela de risco) =====
            if stats_table:
                print("\n")
                print("─" * 80)
                print(f"Estatísticas Descritivas por {hue}: {feature}")
                print("─" * 80)
                
                # Calcular estatísticas para cada grupo do target
                stats_list = []
                for hue_val in sorted(data[hue].unique()):
                    subset = data[data[hue] == hue_val][feature].dropna()
                    stats = subset.describe()
                    
                    stats_row = {
                        hue: hue_val,
                        'count': int(stats['count']),
                        'mean': round(stats['mean'], 2),
                        'std': round(stats['std'], 2),
                        'min': round(stats['min'], 2),
                        '25%': round(stats['25%'], 2),
                        '50%': round(stats['50%'], 2),
                        '75%': round(stats['75%'], 2),
                        'max': round(stats['max'], 2)
                    }
                    stats_list.append(stats_row)
                
                stats_df = pd.DataFrame(stats_list)
                stats_tables[feature] = stats_df
                
                display(stats_df)
                print("\n")
            
            # ===== TABELA DE RISCO =====
            if table:
                series = data[feature].dropna().to_numpy()
                
                # Construir edges (mesma lógica)
                custom_edges = None
                if step is not None:
                    smin = series.min() if start is None else start
                    smax = series.max()

                    bin_start = head_cutoff if head_cutoff is not None else smin
                    
                    if tail_cutoff is not None:
                        bin_end = tail_cutoff
                    elif max_edge is not None:
                        bin_end = max_edge
                    else:
                        bin_end = smax

                    import math
                    finite_edges = [float(bin_start)]
                    if step <= 0:
                        step = 1.0

                    current = float(bin_start)
                    while current < float(bin_end):
                        current += float(step)
                        finite_edges.append(current)

                    left = (-np.inf,) if head_cutoff is not None else ()
                    right = (np.inf,) if tail_cutoff is not None else ()
                    custom_edges = np.array(left + tuple(finite_edges) + right, dtype=float)
                
                if custom_edges is not None:
                    edges = custom_edges
                else:
                    if bins == 'auto':
                        n_bins = min(20, int(np.ceil(np.log2(len(series)) + 1)))
                        _, edges = np.histogram(series, bins=n_bins)
                    else:
                        _, edges = np.histogram(series, bins=bins)
                
                right_flag = False if wrap_cycle is not None else True
                
                data_copy = data.copy()
                data_copy['__bin'] = pd.cut(data_copy[feature], bins=edges, include_lowest=True, right=right_flag)
                
                # Criar labels
                labels = []
                for interval in data_copy['__bin'].cat.categories:
                    lo, hi = interval.left, interval.right
                    if np.isneginf(lo):
                        label = f"<{round(hi, decimal_places)}"
                    elif np.isposinf(hi):
                        label = f">{round(lo, decimal_places)}"
                    else:
                        if wrap_cycle is not None and float(int(lo)) == lo and float(int(hi)) == hi:
                            lo_i = int(lo) % wrap_cycle
                            hi_i = int(hi) % wrap_cycle
                            label = f"{lo_i}-{hi_i}"
                        else:
                            lo_str = _fmt_num(lo, decimal_places)
                            hi_str = _fmt_num(hi, decimal_places)
                            label = f"{lo_str}-{hi_str}"
                    labels.append(label)
                
                label_map = dict(zip(data_copy['__bin'].cat.categories, labels))
                data_copy['__bin_label'] = data_copy['__bin'].map(label_map)
                
                grouped = data_copy.groupby('__bin_label', observed=False, dropna=False).agg(
                    N_Transacoes=(hue, 'count'),
                    N_Target=(hue, 'sum')
                ).reset_index()
                
                grouped.columns = [feature, 'N Transações', f'N {hue.capitalize()}']
                
                # Calcular taxas
                total_target = data[hue].sum()
                total_trans = len(data)
                taxa_geral = total_target / total_trans if total_trans > 0 else 0
                
                grouped[f'Taxa {hue.capitalize()}'] = (grouped[f'N {hue.capitalize()}'] / grouped['N Transações'] * 100).round(1).astype(str) + '%'
                grouped['Risco Relat.'] = grouped[f'N {hue.capitalize()}'] / grouped['N Transações'] / taxa_geral if taxa_geral > 0 else 0
                grouped['Risco Relat.'] = grouped['Risco Relat.'].round(2).astype(str) + 'x'
                
                # Adicionar linha TOTAL
                total_row = pd.DataFrame({
                    feature: ['TOTAL'],
                    'N Transações': [total_trans],
                    f'N {hue.capitalize()}': [total_target],
                    f'Taxa {hue.capitalize()}': [f"{taxa_geral*100:.1f}%"],
                    'Risco Relat.': ['1.00x']
                })
                
                grouped = pd.concat([grouped, total_row], ignore_index=True)
                risk_tables[feature] = grouped
                
                print("─" * 80)
                print(f"Tabela de Análise de Risco: {feature}")
                print("─" * 80)
                display(grouped)
                print("\n")
        
        # Retornar tabelas
        result = {}
        
        if table:
            is_single = isinstance(features, str)
            feat_list = [features] if is_single else list(features)

            if is_single:
                df = risk_tables.get(features)
                if df is not None:
                    df.name = table_name if table_name else f"risk_table_{features}"
                result['risk_table'] = df
            else:
                out = {}
                for f in feat_list:
                    if f in risk_tables:
                        df = risk_tables[f]
                        df.name = f"risk_table_{f}"
                        out[f] = df
                result['risk_tables'] = out
        
        if stats_table:
            is_single = isinstance(features, str)
            feat_list = [features] if is_single else list(features)

            if is_single:
                df = stats_tables.get(features)
                if df is not None:
                    df.name = stats_name if stats_name else f"stats_table_{features}"
                result['stats_table'] = df
            else:
                out = {}
                for f in feat_list:
                    if f in stats_tables:
                        df = stats_tables[f]
                        df.name = f"stats_table_{f}"
                        out[f] = df
                result['stats_tables'] = out
        
        return result if result else None
    
    except Exception as e:
        raise CustomException(e, sys)


def cat_binary_analysis(
    data,
    features,
    hue,
    barplot=True,
    table=True,
    color='#27457B',
    figsize=None,
    category_order=None,
    table_name=None
):
    """
    Análise bivariada para variáveis CATEGÓRICAS (sem bins).
    
    Args:
        data (DataFrame): DataFrame com os dados.
        features (list|str): Lista de features categóricas para análise.
        hue (str): Nome da coluna target binária (0/1).
        barplot (bool): Exibir barplot duplo (freq + taxa target). Default True.
        table (bool): Exibir tabela de análise de risco. Default True.
        color (str): Cor base dos gráficos.
        figsize (tuple): Tamanho da figura.
        category_order (dict): Dicionário com ordem das categorias {feature: [cat1, cat2, ...]}.
        table_name (str): Nome customizado para a tabela.
    
    Returns:
        DataFrame ou dict: DataFrame (uma feature) ou dict de DataFrames (múltiplas), ou None.
    """
    
    try:
        if isinstance(features, str):
            features = [features]
        
        risk_tables = {}
        
        for feature in features:
            show_header = barplot
            if show_header:
                print(f"\n{'─'*80}")
                print(f"Análise Bivariada Categórica: {feature}")
                print(f"{'─'*80}\n")
            
            # ===== BARPLOT DUPLO =====
            if barplot:
                if figsize is None:
                    actual_figsize = (4.27, 3.2)
                else:
                    actual_figsize = figsize
                
                fig, ax = plt.subplots(1, 1, figsize=actual_figsize)
                
                TITLE_SIZE = 8
                TICK_SIZE = 6
                VALUE_SIZE = 6
                LEGEND_SIZE = 6
                TEXT_OFFSET = 0.5
                VALUE_DECIMALS = 1
                
                # Agrupar dados
                data_grouped = (
                    data.groupby([feature])[[feature]]
                        .count().rename(columns={feature: 'count'}).reset_index()
                )
                total = data_grouped['count'].sum()
                total = total if total > 0 else 1
                data_grouped['pct'] = data_grouped['count'] / total * 100.0
                
                # Taxa do target
                target_grouped = (
                    data.groupby([feature])[[hue]].mean().reset_index()
                        .rename(columns={hue: 'taxa_target'})
                )
                target_grouped['taxa_target'] = target_grouped['taxa_target'] * 100.0
                
                data_grouped = data_grouped.merge(target_grouped, on=feature, how='left')
                
                # Aplicar ordenação customizada se fornecida
                if category_order and feature in category_order:
                    order = category_order[feature]
                    order = [cat for cat in order if cat in data_grouped[feature].values]
                    data_grouped[feature] = pd.Categorical(data_grouped[feature], categories=order, ordered=True)
                    data_grouped = data_grouped.sort_values(feature).reset_index(drop=True)
                
                # Posições X
                x_positions = np.arange(len(data_grouped))
                bar_width = 0.35
                
                # Barras de frequência
                ax.bar(
                    x=x_positions - bar_width/2,
                    height=data_grouped['pct'],
                    width=bar_width,
                    color=color,
                    label='Freq. Relativa (%)'
                )
                
                for idx, v in enumerate(data_grouped['pct']):
                    ax.text(x_positions[idx] - bar_width/2, v + TEXT_OFFSET, 
                           f'{v:.{VALUE_DECIMALS}f}%',
                           ha='center', fontsize=VALUE_SIZE)
                
                # Barras da taxa target
                ax.bar(
                    x=x_positions + bar_width/2,
                    height=data_grouped['taxa_target'],
                    width=bar_width,
                    color='#e85d04',
                    label=f'Taxa {hue} (%)'
                )
                
                for idx, v in enumerate(data_grouped['taxa_target']):
                    ax.text(x_positions[idx] + bar_width/2, v + TEXT_OFFSET, 
                           f'{v:.{VALUE_DECIMALS}f}%',
                           ha='center', fontsize=VALUE_SIZE)
                
                # Configurações
                labels = data_grouped[feature].tolist()
                ax.set_xticks(x_positions)
                ax.set_xticklabels(labels, fontsize=TICK_SIZE)
                ax.get_yaxis().set_visible(False)
                for sp in ('top', 'right', 'bottom', 'left'):
                    ax.spines[sp].set_visible(False)
                ax.grid(False)
                ax.set_title(feature, fontsize=TITLE_SIZE)
                ax.set_xlabel("", fontsize=TITLE_SIZE-3)
                ax.set_ylabel("", fontsize=TITLE_SIZE-3)
                ax.legend(loc='best', fontsize=LEGEND_SIZE, frameon=False)
                
                plt.tight_layout()
                plt.show()
            
            # ===== TABELA DE RISCO =====
            if table:
                grouped = data.groupby(feature, dropna=False).agg(
                    N_Transacoes=(hue, 'count'),
                    N_Target=(hue, 'sum')
                ).reset_index()
                
                grouped.columns = [feature, 'N Transações', f'N {hue.capitalize()}']
                grouped = grouped[grouped['N Transações'] > 0].copy()
                
                # Aplicar ordenação customizada se fornecida
                if category_order and feature in category_order:
                    order = category_order[feature]
                    order = [cat for cat in order if cat in grouped[feature].values]
                    grouped[feature] = pd.Categorical(grouped[feature], categories=order, ordered=True)
                    grouped = grouped.sort_values(feature).reset_index(drop=True)
                
                # Calcular taxas
                total_target = data[hue].sum()
                total_trans = len(data)
                taxa_geral = total_target / total_trans if total_trans > 0 else 0
                
                grouped[f'Taxa {hue.capitalize()}'] = (grouped[f'N {hue.capitalize()}'] / grouped['N Transações'] * 100).round(1).astype(str) + '%'
                grouped['Risco Relat.'] = grouped[f'N {hue.capitalize()}'] / grouped['N Transações'] / taxa_geral if taxa_geral > 0 else 0
                grouped['Risco Relat.'] = grouped['Risco Relat.'].round(2).astype(str) + 'x'
                
                # Adicionar linha TOTAL
                total_row = pd.DataFrame({
                    feature: ['TOTAL'],
                    'N Transações': [total_trans],
                    f'N {hue.capitalize()}': [total_target],
                    f'Taxa {hue.capitalize()}': [f"{taxa_geral*100:.1f}%"],
                    'Risco Relat.': ['1.00x']
                })
                
                grouped = pd.concat([grouped, total_row], ignore_index=True)
                risk_tables[feature] = grouped
                
                print(f"\n{'─'*80}")
                print(f"Tabela de Análise de Risco: {feature}")
                print(f"{'─'*80}\n")
                display(grouped)
                print("\n")
        
        # Retornar tabelas
        if table:
            is_single = isinstance(features, str)
            feat_list = [features] if is_single else list(features)

            if is_single:
                df = risk_tables.get(features)
                if df is not None:
                    df.name = table_name if table_name else f"risk_table_{features}"
                return df
            else:
                out = {}
                for f in feat_list:
                    if f in risk_tables:
                        df = risk_tables[f]
                        df.name = f"risk_table_{f}"
                        out[f] = df
                return out
        else:
            return None
    
    except Exception as e:
        raise CustomException(e, sys)
        
def consolidate_risk_tables(tables, sort_by='relative_risk', ascending=False):
    """
    Consolida múltiplas tabelas de risco geradas pela função binary_analysis.
    
    Args:
        tables (list): Lista contendo DataFrames ou dicionários de DataFrames 
                      retornados pela binary_analysis.
        sort_by (str): Coluna para ordenação. Default: 'relative_risk'.
        ascending (bool): Ordem crescente ou decrescente. Default: False (maior→menor).
    
    Returns:
        pd.DataFrame: Tabela consolidada com colunas padronizadas e prioridade.
    
    Exemplo:
        >>> rt = binary_analysis(data=df, features='transaction_amount', 
                                 hue='has_cbk', start=0, step=250, 
                                 tail_cutoff=1500, table=True)
        >>> result = binary_analysis(data=df, features=['age', 'balance'], 
                                     hue='has_cbk', table=True)
        >>> consolidated = consolidate_risk_tables(tables=[rt, result])
    """
    try:
        consolidated_data = []
        
        for table_input in tables:
            # Caso 1: É um DataFrame individual
            if isinstance(table_input, pd.DataFrame):
                tables_to_process = [table_input]
                feature_names = [table_input.name if hasattr(table_input, 'name') else 'unknown_feature']
            
            # Caso 2: É um dicionário de DataFrames
            elif isinstance(table_input, dict):
                tables_to_process = list(table_input.values())
                feature_names = list(table_input.keys())
            
            else:
                print(f"Warning: tipo não suportado {type(table_input)}. Pulando...")
                continue
            
            # Processar cada tabela
            for df, feature_name in zip(tables_to_process, feature_names):
                if df is None or df.empty:
                    continue
                
                # Identificar colunas dinamicamente
                cols = df.columns.tolist()
                
                # Primeira coluna é sempre a feature/bin
                bin_col = cols[0]
                
                # Identificar coluna do target (ex: 'N Has_cbk', 'N Fraude')
                target_col = [c for c in cols if c.startswith('N ') and c != 'N Transações'][0]
                target_name = target_col.replace('N ', '').lower()
                
                # Identificar coluna de taxa (ex: 'Taxa Has_cbk', 'Taxa Fraude')
                rate_col = [c for c in cols if c.startswith('Taxa ')][0]
                
                # Processar cada linha (exceto TOTAL)
                for idx, row in df.iterrows():
                    bin_value = row[bin_col]
                    
                    # Pular linha TOTAL
                    if str(bin_value).upper() == 'TOTAL':
                        continue
                    
                    # Extrair valores
                    n_total = row['N Transações']
                    n_target = row[target_col]
                    
                    # Limpar taxa (remover '%')
                    # target_rate = float(str(row[rate_col]).replace('%', ''))
                    
                    # Manter taxa com formatação (%)
                    target_rate = str(row[rate_col])
                    
                    # Limpar risco relativo (remover 'x')
                    relative_risk = float(str(row['Risco Relat.']).replace('x', ''))
                    
                    # Definir prioridade
                    if relative_risk >= 2.5:
                        priority = 'CRÍTICA'
                    elif relative_risk >= 2.0:
                        priority = 'ALTA'
                    elif relative_risk >= 1.5:
                        priority = 'MÉDIA'
                    else:
                        priority = 'BAIXA'
                    
                    consolidated_data.append({
                        'feature': feature_name,
                        'bin': str(bin_value),
                        'n_total': n_total,
                        f'n_{target_name}': n_target,
                        f'{target_name}_rate': target_rate,
                        'relative_risk': relative_risk,
                        'priority': priority
                    })
        
        # Criar DataFrame consolidado
        consolidated_df = pd.DataFrame(consolidated_data)
        
        # Ordenar
        if sort_by in consolidated_df.columns:
            consolidated_df = consolidated_df.sort_values(
                by=sort_by, 
                ascending=ascending
            ).reset_index(drop=True)
        
        return consolidated_df
    
    except Exception as e:
        raise CustomException(e, sys)


# Exemplo de uso adicional com visualização
def display_risk_summary(consolidated_df, top_n=10):
    """
    Exibe um resumo visual das principais features de risco.
    
    Args:
        consolidated_df (pd.DataFrame): Tabela consolidada de riscos.
        top_n (int): Número de bins de maior risco a exibir. Default: 10.
    """
    try:
        print(f"\n{'='*80}")
        print(f"RESUMO DE ANÁLISE DE RISCO - TOP {top_n} BINS")
        print(f"{'='*80}\n")
        
        # Filtrar top N
        top_risks = consolidated_df.head(top_n).copy()
        
        # Exibir tabela
        display(top_risks)
        
        # Estatísticas por prioridade
        print(f"\n{'─'*80}")
        print("DISTRIBUIÇÃO POR PRIORIDADE")
        print(f"{'─'*80}\n")
        
        priority_summary = consolidated_df.groupby('priority').agg({
            'feature': 'count',
            'relative_risk': 'mean'
        }).round(2)
        priority_summary.columns = ['Quantidade de Bins', 'Risco Relativo Médio']
        
        # Ordenar por ordem de severidade
        priority_order = ['CRÍTICA', 'ALTA', 'MÉDIA', 'BAIXA']
        priority_summary = priority_summary.reindex(
            [p for p in priority_order if p in priority_summary.index]
        )
        
        display(priority_summary)
        
        # Features com maior risco médio
        print(f"\n{'─'*80}")
        print("FEATURES COM MAIOR RISCO MÉDIO")
        print(f"{'─'*80}\n")
        
        feature_risk = consolidated_df.groupby('feature').agg({
            'relative_risk': 'mean',
            'bin': 'count'
        }).round(2).sort_values('relative_risk', ascending=False)
        feature_risk.columns = ['Risco Relativo Médio', 'Número de Bins']
        
        display(feature_risk)
        
    except Exception as e:
        raise CustomException(e, sys)
        
def frequency_tables(
    data,
    features=None,
    abs_freq=True,
    rel_freq=True,
    cum_freq=True,
    bins='auto',
    decimal_places=2,
    show=True,
    step=None,
    start=None,
    head_cutoff=None,
    tail_cutoff=None,
    max_edge=None,      # força última borda (ex.: 24 p/ horas)
    wrap_cycle=None     # rótulos circulares (ex.: 23–0 p/ horas)
):
    """
    Gera tabelas de frequência com rótulos inteligentes e bins configuráveis.

    Parâmetros:
        data (pd.DataFrame): base de dados.
        features (list|str): colunas numéricas a analisar.
        abs_freq (bool): incluir frequência absoluta.
        rel_freq (bool): incluir frequência relativa.
        cum_freq (bool): incluir frequência acumulada.
        bins (int|str): nº ou estratégia padrão de bins.
        decimal_places (int): casas decimais nos rótulos/porcentagens.
        show (bool): exibir resultado com display(HTML()).
        step (float): largura fixa dos bins (ex.: 900).
        start (float): valor inicial dos bins.
        head_cutoff (float): cria bin inicial "< head".
        tail_cutoff (float): cria bin final "> tail".
        max_edge (float): força última borda (ex.: 24 para hora).
        wrap_cycle (int): torna rótulos circulares (ex.: 23–0 para hora).

    Retorna:
        dict: dicionário {coluna: tabela de frequências}
    """
    try:
        from IPython.display import display, HTML

        if features is None:
            features = data.select_dtypes(include=[np.number]).columns.tolist()
        if isinstance(features, str):
            features = [features]

        tables_dict = {}

        for feature in features:
            if feature not in data.columns:
                print(f"Warning: coluna '{feature}' não encontrada. Pulando...")
                continue
            if not pd.api.types.is_numeric_dtype(data[feature]):
                print(f"Warning: coluna '{feature}' não é numérica. Pulando...")
                continue

            series = data[feature].dropna().to_numpy()

            # --- construir edges ---
            custom_edges = None
            if step is not None:
                smin = series.min() if start is None else start
                smax = series.max()
                top = max_edge if max_edge is not None else (
                    tail_cutoff if tail_cutoff is not None else smax
                )
                left = (-np.inf,) if head_cutoff is not None else ()
                right = (np.inf,) if tail_cutoff is not None else ()
                edges_mid = np.arange(float(smin), float(top) + float(step), float(step))
                custom_edges = np.array(left + tuple(edges_mid) + right, dtype=float)
            elif (head_cutoff is not None) or (tail_cutoff is not None):
                _, base_edges = np.histogram(series, bins=bins)
                left = (-np.inf,) if head_cutoff is not None else ()
                right = (np.inf,) if tail_cutoff is not None else ()
                custom_edges = np.array(left + tuple(base_edges) + right, dtype=float)

            if custom_edges is not None:
                edges = custom_edges
                counts, _ = np.histogram(series, bins=edges)
            else:
                counts, edges = np.histogram(series, bins=bins)

            # --- rótulos + bordas salvas ---
            labels = []
            lowers = []
            uppers = []
            for i in range(len(edges) - 1):
                lo, hi = edges[i], edges[i + 1]
                lowers.append(lo); uppers.append(hi)
                if np.isneginf(lo):
                    label = f"<{round(hi, decimal_places)}"
                elif np.isposinf(hi):
                    label = f">{round(lo, decimal_places)}"
                else:
                    if wrap_cycle is not None and float(int(lo)) == lo and float(int(hi)) == hi:
                        lo_i = int(lo) % wrap_cycle
                        hi_i = int(hi) % wrap_cycle
                        label = f"{lo_i}-{hi_i}"
                    else:
                        label = f"{round(lo, decimal_places)}-{round(hi, decimal_places)}"
                labels.append(label)

            freq_table = pd.DataFrame({
                feature: labels,
                'Frequência Absoluta': counts,
                '__lower_edge': lowers,      # <<< borda esquerda numérica
                '__upper_edge': uppers       # <<< borda direita numérica
            })

            if abs_freq and rel_freq:
                pct = freq_table['Frequência Absoluta'] / max(freq_table['Frequência Absoluta'].sum(), 1)
                freq_table['Frequência Relativa'] = (pct * 100).round(decimal_places).astype(str) + '%'

            if cum_freq:
                cpct = freq_table['Frequência Absoluta'].cumsum() / max(freq_table['Frequência Absoluta'].sum(), 1)
                freq_table['Frequência Acumulada'] = (cpct * 100).round(decimal_places).astype(str) + '%'

            tables_dict[feature] = freq_table

        if show:
            for feat, tb in tables_dict.items():
                tb_display = tb.copy()
                tb_display = tb_display.drop(columns=[c for c in tb_display.columns if c.startswith('__')], errors='ignore')
                
                display(HTML(f"<h4 style='margin:8px 0'>{feat}</h4>"))
                display(tb_display)

        return tables_dict

    except Exception as e:
        raise CustomException(e, sys)

def plot_binary_distribution(
    df: pd.DataFrame,
    target_column: str,
    title: Optional[str] = None,
    labels: Optional[Dict[int, str]] = None,
    colors: Optional[Dict[int, str]] = None,
    figsize: Tuple[float, float] = (2.5, 2.5),
    title_fontsize: int = 8,
    label_fontsize: int = 8,
    annotation_fontsize: int = 8,
    annotation_color: str = 'white',
    show_percentage: bool = True,
    sort_by_value: bool = True,
    reverse_order: bool = False,
    title_position: str = 'left',
    return_fig: bool = False,
    save_path: Optional[str] = None
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Gera um gráfico de barras mostrando a distribuição de uma variável binária.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contendo os dados
    target_column : str
        Nome da coluna binária (0/1)
    title : str, optional
        Título do gráfico. Se None, usa padrão genérico
    labels : dict, optional
        Dicionário mapeando valores para labels {0: 'Label0', 1: 'Label1'}
        Default: {0: 'Classe 0', 1: 'Classe 1'}
    colors : dict, optional
        Dicionário mapeando valores para cores {0: '#cor0', 1: '#cor1'}
        Default: {0: '#023047', 1: '#e85d04'}
    figsize : tuple, default=(2.5, 2.5)
        Tamanho da figura (largura, altura)
    title_fontsize : int, default=8
        Tamanho da fonte do título
    label_fontsize : int, default=8
        Tamanho da fonte dos labels do eixo X
    annotation_fontsize : int, default=8
        Tamanho da fonte das anotações nas barras
    annotation_color : str, default='white'
        Cor do texto das anotações
    show_percentage : bool, default=True
        Se True, mostra percentuais nas barras
    sort_by_value : bool, default=True
        Se True, ordena barras por valor (menor para maior)
    reverse_order : bool, default=False
        Se True, inverte a ordem das barras no eixo X
    title_position : str, default='left'
        Posição do título ('left', 'center', 'right')
    return_fig : bool, default=False
        Se True, retorna (fig, ax) ao invés de mostrar o gráfico
    save_path : str, optional
        Se fornecido, salva o gráfico no caminho especificado
    
    Returns
    -------
    tuple or None
        Se return_fig=True, retorna (fig, ax). Caso contrário, None.
    
    Examples
    --------
    >>> # Análise de fraude
    >>> plot_binary_distribution(
    ...     df, 
    ...     target_column='has_cbk',
    ...     title='A taxa de fraude é aproximadamente 12.2%',
    ...     labels={0: 'Sem fraude', 1: 'Fraude'},
    ...     colors={0: '#023047', 1: '#e85d04'}
    ... )
    
    >>> # Análise de churn
    >>> plot_binary_distribution(
    ...     df,
    ...     target_column='churned',
    ...     title='Taxa de churn do trimestre',
    ...     labels={0: 'Ativo', 1: 'Churn'},
    ...     colors={0: '#2ecc71', 1: '#e74c3c'}
    ... )
    
    >>> # Análise de conversão
    >>> plot_binary_distribution(
    ...     df,
    ...     target_column='converted',
    ...     title='Taxa de conversão da campanha',
    ...     labels={0: 'Não converteu', 1: 'Converteu'},
    ...     colors={0: '#95a5a6', 1: '#3498db'}
    ... )
    """
    
    # Validações
    if target_column not in df.columns:
        raise ValueError(f"Coluna '{target_column}' não encontrada no DataFrame")
    
    unique_values = df[target_column].dropna().unique()
    if len(unique_values) != 2:
        raise ValueError(f"Coluna '{target_column}' deve ter exatamente 2 valores únicos. Encontrados: {len(unique_values)}")
    
    if not all(val in [0, 1] for val in unique_values):
        raise ValueError(f"Coluna '{target_column}' deve conter apenas valores 0 e 1. Valores encontrados: {unique_values}")
    
    # Configurações padrão
    if labels is None:
        labels = {0: 'Classe 0', 1: 'Classe 1'}
    
    if colors is None:
        colors = {0: '#023047', 1: '#e85d04'}
    
    if title_position not in ['left', 'center', 'right']:
        raise ValueError("title_position deve ser 'left', 'center' ou 'right'")
    
    # Agrupa os dados
    distribution = df.groupby([target_column])[[target_column]].count()
    distribution = distribution.rename(columns={target_column: 'count'}).reset_index()
    distribution['pct'] = (distribution['count'] / distribution['count'].sum()) * 100
    
    if sort_by_value:
        distribution = distribution.sort_values(by=['pct'])
    
    # Calcula título automaticamente se não fornecido
    if title is None:
        if 1 in distribution[target_column].values:
            positive_rate = distribution[distribution[target_column] == 1]['pct'].values[0]
            title = f'Taxa da classe positiva: {positive_rate:.1f}%'
        else:
            title = 'Distribuição da variável binária'
    
    # Prepara cores e labels na ordem correta
    bar_colors = [colors[val] for val in distribution[target_column]]
    x_labels = [labels[val] for val in distribution[target_column]]
    
    # Cria a figura
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plota as barras
    bars = ax.bar(
        x=range(len(distribution)), 
        height=distribution['pct'], 
        color=bar_colors
    )
    
    # Customiza o plot
    ax.set_title(
        title, 
        fontweight='bold', 
        fontsize=title_fontsize, 
        pad=15, 
        loc=title_position
    )
    ax.set_xlabel('')
    ax.set_xticks(ticks=range(len(distribution)), labels=x_labels, fontsize=label_fontsize)
    ax.tick_params(axis='both', which='both', length=0)
    
    if reverse_order:
        ax.invert_xaxis()
    
    # Remove elementos desnecessários
    ax.yaxis.set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(False)
    
    # Adiciona anotações nas barras
    if show_percentage:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f'{height:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, -8),
                textcoords="offset points",
                ha='center', 
                va='center',
                fontsize=annotation_fontsize, 
                color=annotation_color
            )
    
    plt.tight_layout()
    
    # Salva o gráfico se caminho fornecido
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if return_fig:
        return fig, ax
    else:
        plt.show()
        return None


# ============================================
# EXEMPLOS DE USO
# ============================================

if __name__ == "__main__":
    
    # Exemplo 1: Análise de fraude (seu caso original)
    plot_binary_distribution(
        df,
        target_column='has_cbk',
        title='A taxa de fraude é aproximadamente 12.2%',
        labels={0: 'Sem fraude', 1: 'Fraude'},
        colors={0: '#023047', 1: '#e85d04'},
        reverse_order=True
    )
    
    # Exemplo 2: Análise de churn
    plot_binary_distribution(
        df,
        target_column='churned',
        title='Taxa de churn no último trimestre',
        labels={0: 'Cliente Ativo', 1: 'Churn'},
        colors={0: '#2ecc71', 1: '#e74c3c'},
        figsize=(3, 3)
    )
    
    # Exemplo 3: Análise de conversão
    plot_binary_distribution(
        df,
        target_column='converted',
        title='Performance da Campanha de Marketing',
        labels={0: 'Não Converteu', 1: 'Converteu'},
        colors={0: '#95a5a6', 1: '#27ae60'}
    )
    
    # Exemplo 4: Análise de aprovação de crédito
    plot_binary_distribution(
        df,
        target_column='approved',
        title='Taxa de aprovação de crédito',
        labels={0: 'Negado', 1: 'Aprovado'},
        colors={0: '#c0392b', 1: '#16a085'}
    )
    
    # Exemplo 5: Análise de inadimplência
    plot_binary_distribution(
        df,
        target_column='defaulted',
        title='Taxa de inadimplência da carteira',
        labels={0: 'Adimplente', 1: 'Inadimplente'},
        colors={0: '#3498db', 1: '#e67e22'}
    )
    
    # Exemplo 6: Sem título customizado (usa padrão)
    plot_binary_distribution(
        df,
        target_column='target',
        labels={0: 'Negativo', 1: 'Positivo'}
    )
    
    # Exemplo 7: Salvando o gráfico
    plot_binary_distribution(
        df,
        target_column='has_cbk',
        title='Análise de Fraude',
        labels={0: 'Sem fraude', 1: 'Fraude'},
        save_path='fraud_analysis.png'
    )
    
    # Exemplo 8: Retornando fig e ax para customizações extras
    fig, ax = plot_binary_distribution(
        df,
        target_column='has_cbk',
        title='Taxa de fraude',
        labels={0: 'Sem fraude', 1: 'Fraude'},
        return_fig=True
    )
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('#ffffff')
    plt.show()