"""
Программа чтения данных по заданному пути
"""


from typing import Text
import pandas as pd

def get_dataset(dataset_path: Text) -> pd.DataFrame:
    """
    Получение данных по заданному пути
    :param dataset_path: путь до данных
    :return: возвращает датасет
    """
    return pd.read_csv(dataset_path)