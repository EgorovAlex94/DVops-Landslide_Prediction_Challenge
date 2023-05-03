import os
import yaml
import streamlit as st
from src.data.get_data import load_data, get_dataset
from src.evaluate.evaluate import evaluate_from_file

CONFIG_PATH = "../config/params.yml"


st.markdown("# Prediction")
st.sidebar.markdown("# Prediction")

with open(CONFIG_PATH) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
endpoint = config["endpoints"]["prediction_from_file"]

upload_file = st.file_uploader(
    "", type=["csv", "xlsx"], accept_multiple_files=False
)
# Проверка загружен ли файл
if upload_file:
    dataset_csv_df, files = load_data(data=upload_file, type_data="Test")
    # проверка на наличие сохраненной модели
    if os.path.exists(config["train"]["model_path"]):
        evaluate_from_file(data=dataset_csv_df, endpoint=endpoint, files=files)
    else:
        st.error("Сначала обучите модель")