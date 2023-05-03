"""
Программа: Отрисовка графиков
"""


import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns



def barplot_group(
        data: pd.DataFrame, col_main: str, col_group: str, title: str
) -> matplotlib.figure.Figure:
    """
    Отрисовка графика boxplot
    :param data: датасет
    :param col_main: признак для анализа по col_group
    :param col_group: признак для группировки
    :param title: название графика
    :return: поле рисунка
    """
    data_group = (
        data.groupby([col_group])[col_main]
        .value_counts(normalize=True)
        .rename("percantage")
        .mul(100)
        .reset_index()
        .sort_values(col_group)
    )

    data_group.columns = [col_group, col_main, "percantage"]


    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(15, 7))

    ax = sns.barplot(
        x=col_main, y="percantage", hue=col_group, data=data_group, palette="rocket"
    )
    for patch in ax.patches:
        percentage = "{:.1f}%".format(patch.get_height())
        ax.annotate(percentage,  # текст
                    # координата xy
                    (patch.get_x() + patch.get_width() / 2.,
                     patch.get_height()),
                    # центрирование
                    ha='center',
                    va='center',
                    xytext=(0, 10),
                    # точка смещения относительно координаты
                    textcoords='offset points',
                    fontsize=14,
                    )
    plt.title(title, fontsize=20)
    plt.xlabel(col_main, fontsize=14)
    plt.ylabel('percantage', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    return fig

def heatmap(
        data: pd.DataFrame, cols: str, title: str
) -> matplotlib.figure.Figure:
    """
    Отрисовка графика heatmap
    :param data: датасет
    :param col_group: признаки для корреляции
    :param title: название графика
    :return: поле рисунка
    """
    fig = plt.figure(figsize=(10, 10))
    sns.heatmap(data[cols].corr(method='spearman'), annot=True, fmt='.2f')
    plt.title('Корреляция признаков', fontsize=20)
    return fig

def norm_target(data: pd.DataFrame, col: str, title: str
) -> matplotlib.figure.Figure:
    """
    Отрисовка графика дисбаланс классов
    :param data: датасет
    :param col: признак
    :param title: газвание графика
    :return: поле рисунка
    """
    data_group = (
        data
        .Label
        .value_counts(normalize=True)
        .mul(100)
        .rename('percent')
        .reset_index()
    )

    fig = plt.figure(figsize=(15, 7))

    ax = sns.barplot(
        x="index", y="percent",  data=data_group
    )
    for patch in ax.patches:
        percentage = "{:.1f}%".format(patch.get_height())
        ax.annotate(percentage,  # текст
                    # координата xy
                    (patch.get_x() + patch.get_width() / 2.,
                     patch.get_height()),
                    # центрирование
                    ha='center',
                    va='center',
                    xytext=(0, 10),
                    # точка смещения относительно координаты
                    textcoords='offset points',
                    fontsize=14,
                    )
    plt.title(title, fontsize=20)
    plt.xlabel(col, fontsize=14)
    plt.ylabel('percantage', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    return fig
