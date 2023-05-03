"""
Программа: Модель для прогназирования того, произойдет ли оползень на определенном выбранном участке 25мх25м
"""


import warnings
import optuna
import pandas as pd

import uvicorn
from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile
from pydantic import BaseModel

from src.pipelines.pipeline import pipeline_training
from src.evaluate.evaluate import piplene_evaluate
from src.train.metrics import load_metrics

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

app = FastAPI()
CONFIG_PATH = "../config/params.yml"

#class GeographicalData(BaseModel)
#"""
#    Признаки для получения результатов модели(топ 20, остальные константа)
#"""

@app.get("/hello")
def welcome():
    """
    Hello
    :return: None
    """
    return {'message': 'Hello Data Scientist!'}


@app.post("/train")
def training():
    """
    Обучение модели
    """
    pipeline_training(config_path=CONFIG_PATH)
    metrics = load_metrics(config_path=CONFIG_PATH)

    return {"metrics": metrics}

@app.post("/predict")
def prediction(file: UploadFile = File(...)):
    """
    Предсказание модели по данным из файла
    :param file:
    :return:
    """
    result = piplene_evaluate(config_path=CONFIG_PATH, data_path=file.file)
    assert isinstance(result, list), "Результат не соотвествует типу list"
    return {"prediction": result[:5]}


#@app.post("/predict_input")
#def predict_input(customer: InsuranceCustomer):
    """
    Предсказания модели по введенным данным
    """
"""
   features = [
       [

        ]
    ]

    cols =

    data = pd.DataFrame(features, columns=cols
    prediction = pipeline_evaluate(congig_path=CONFIG_PATH, dataset=data)[0]
    result = (
        {"This area is subject to a landslide"}
        if prediction == 1
        else {"This area is not subject to a landslide"}
        if prediction == 0
        else "Error"
    )
    return result
"""


if __name__ == "__main__":
    #Запуск сервера, используя хост и порт
    uvicorn.run(app, host="127.0.0.1", port=80)


