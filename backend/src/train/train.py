"""
Программа: Тренировка данных
Версия 1.0
"""

import optuna
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, \
StratifiedShuffleSplit, GridSearchCV, RandomizedSearchCV, cross_val_score
from optuna import Study
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from ..data.split_dataset import get_train_test_data
from ..train.metrics import save_metrics

def objective(trial, X, y, N_FOLDS, random_state=10) -> np.array:
    """
    :param trial: кол-во trials
    :param X: данные объект-признаки
    :param y: данные целевая переменная
    :param N_FOLDS: кол-во фолдов
    :param random_state: random_state
    :return: среднее значение метрики по фолдам
    """
    lgb_params = {
        "n_estimators":
        trial.suggest_categorical("n_estimators", [1000]),
        "learning_rate":
        trial.suggest_categorical("learning_rate", [0.013607708322422411]),
        "num_leaves":
        trial.suggest_int("num_leaves", 20, 1000, step=20),
        "max_depth":
        trial.suggest_int("max_depth", 3, 12),
        "lambda_l1":
        trial.suggest_int("lambda_l1", 0, 100),
        "lambda_l2":
        trial.suggest_int("lambda_l2", 0, 100),
        "min_gain_to_split":
        trial.suggest_int("min_gain_to_split", 0, 20),
        "bagging_fraction":
        trial.suggest_float("bagging_fraction", 0.2, 1.0),
        "bagging_freq":
        trial.suggest_categorical("bagging_freq", [1]),
        "feature_fraction":
        trial.suggest_float("feature_fraction", 0.2, 1.0),
        "random_state": trial.suggest_categorical("random_state", [random_state])
    }

    N_FOLDS = 3
    cv = KFold(n_splits=N_FOLDS, shuffle=True)

    cv_predicts = np.empty(N_FOLDS)
    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # стрижка "на лету"
        # observation_key - Оценочная метрика для обрезки
        pruning_callback = optuna.integration.LightGBMPruningCallback(
            trial, "F1")
        model = LGBMClassifier(**lgb_params)
        model.fit(X_train,
                  y_train,
                  eval_set=[(X_test, y_test)],
                  eval_metric="F1",
                  early_stopping_rounds=100,
                  verbose=0)

        preds = model.predict(X_test)
        cv_predicts[idx] = f1_score(y_test, preds)

    return np.mean(cv_predicts)

def find_optimal_params(
        data_train: pd.DataFrame, data_test: pd.DataFrame, **kwargs
) -> Study:
    """
    :param data_train: датасет train
    :param data_test: датасет test
    :return: [LGBMClassifier tuning, Study]
    """
    x_train, x_test, y_train, y_test = get_train_test_data(
        data_train = data_train, data_test = data_test, target = kwargs["target_column"]
    )

    study = optuna.create_study(direction="maximize", study_name="LGB")
    func = lambda trial: objective(
        trial, x_train, y_train, kwargs['n_folds'], kwargs["random_state"]
    )
    study.optimize(func, n_trials=kwargs["n_trials"], show_progress_bar=True)

    return study

def train_model(
        data_train: pd.DataFrame,
        data_test: pd.DataFrame,
        study: Study,
        target: str,
        metric_path: str
) -> LGBMClassifier:
    """
    :param data_train: тренировочный датасет
    :param data_test: тестовый датасет
    :param study: study optuna
    :param target: название целевой переменной
    :param metric_path: путь до папки с метриками
    :return: LGBMClassifier
    """
    # get data
    x_train, x_test, y_train, y_test = get_train_test_data(
        data_train=data_train, data_test=data_test, target=target
    )

    # training with optimal params
    lgb_grid = LGBMClassifier(**study.best_params, silent=True, verbose=-1)
    lgb_grid.fit(x_train,
                 y_train,
                 eval_metric="F1",
                 eval_set=[(x_test, y_test)],
                 verbose=False,
                 early_stopping_rounds=100)

    # save metrics
    save_metrics(data_x=x_test, data_y=y_test, model=lgb_grid, metric_path=metric_path)

    return lgb_grid