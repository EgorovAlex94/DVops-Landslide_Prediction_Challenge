"""
Программа разделения датасета на тренировочный и тестовый
"""

from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def split_train_test(dataset: pd.DataFrame, **kwargs):
    """
    Разделения датасета на тренировочный и тестовый датасеты
    :param dataset: датасет
    :return: тренировочный и тестовый датасеты
    """
    df_train, df_test = train_test_split(
        dataset,
        test_size=kwargs["test_size"],
        stratify=dataset[kwargs["target_column"]],
        random_state=kwargs["random_state"]
    )
    return df_train, df_test


def get_train_test_data(
        data_train: pd.DataFrame, data_test: pd.DataFrame, target: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Получение train/test с разделением на признаки и таргет
    :param data_train: тренировочный датасет
    :param data_test: тестовый датасет
    :param target: целевая переменная
    :return: набор тестовых и тренировочных данных
    """
    x_train, x_test = (
        data_train.drop(target, axis=1),
        data_test.drop(target, axis=1),
    )
    y_train, y_test = (
        data_train.loc[:, target],
        data_test.loc[:, target],
    )
    return x_train, x_test, y_train, y_test

