import os
import joblib
import yaml
import pandas as pd

from ..data.get_data import get_dataset
from ..transform.transform import FeatureEngineering

def piplene_evaluate(
        config_path, dataset: pd.DataFrame = None, data_path: str = None
) -> list:
    """
    Предобработка входных данных и получение предсказаний
    :param config_path: датасет
    :param dataset: путь до кфг файла
    :param data_path: путь до файла с данными
    :return: предсказания
    """
    # get params
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    train_config = config["train"]

    # preprocessing
    if data_path:
        dataset = get_dataset(dataset_path=data_path)
    s = FeatureEngineering(data=dataset)
    dataset = s.feature_engineering()

    model = joblib.load(os.path.join(train_config["model_path"]))
    prediction = model.predict(dataset).tolist()

    return prediction