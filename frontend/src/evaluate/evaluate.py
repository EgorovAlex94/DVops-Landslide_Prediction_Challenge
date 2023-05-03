import json
from io import BytesIO
import pandas as pd
import requests
import streamlit as st

"""
def evaluate_input(unique_data_path: str, endpoint: object) -> None:
  
    Получение входных данных путем ввода в UI -> вывод результата
    :param unique_data_path: путь до уникальных значений
    :param endpoint: endpoint
    
    with open(unique_data_path) as file:
        unique_df = json.load(file)

        #
"""

def evaluate_from_file(data: pd.DataFrame, endpoint: object, files: BytesIO):
    """
    Получение входных данных в качества файла -> вывод результата в виде таблицы
    :param data: датасет
    :param endpoint: endpoint
    :param files: файл
    :return: Предсказанные значения
    """
    button_ok = st.button("Predict")
    if button_ok:
        # загрушка, чтобы не выводить все предсказания
        data_ = data[:5]
        output = requests.post(endpoint, files=files, timeout=8000)
        data_["predict"] = output.json()["prediction"]
        st.write(data_.head())