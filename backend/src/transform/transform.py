"""
Программа: предобработка данных
Версия 1.0
"""

import json
import warnings
import pandas as pd
import statistics

warnings.filterwarnings("ignore")


class FeatureEngineering:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.df_train = pd.DataFrame()
        self.selectedCols = [
            'elevation', 'lsfactor', 'placurv', 'procurv', 'sdoif', 'slope',
            'twi', 'aspect'
        ]
        self.geology = [
            '1_geology', '2_geology', '3_geology', '4_geology', '5_geology',
            '6_geology', '7_geology', '8_geology', '9_geology', '10_geology',
            '11_geology', '12_geology', '13_geology', '14_geology',
            '15_geology', '16_geology', '17_geology', '18_geology',
            '19_geology', '20_geology', '21_geology', '22_geology',
            '23_geology', '24_geology', '25_geology'
        ]
        self.geology_full = [
            '1_geology', '2_geology', '3_geology', '4_geology', '5_geology',
            '6_geology', '7_geology', '8_geology', '9_geology', '10_geology',
            '11_geology', '12_geology', '13_geology', '14_geology',
            '15_geology', '16_geology', '17_geology', '18_geology',
            '19_geology', '20_geology', '21_geology', '22_geology',
            '23_geology', '24_geology', '25_geology', 'mode_geology'
        ]

    def feature_engineering(self) -> pd.DataFrame:
        for i in self.selectedCols:
            self.df_train[i + "_min"] = self.data[[
                x for x in self.data.columns if i in x
            ]].min(axis=1)
            self.df_train[i + "_max"] = self.data[[
                x for x in self.data.columns if i in x
            ]].max(axis=1)
            self.df_train[i + "_range"] = self.df_train[
                i + "_max"] - self.df_train[i + "_min"]

        self.data['mode_geology'] = self.data.apply(
            lambda x: int(statistics.mode(x[self.geology])), axis=1)
        for col in self.data.columns:
            if col in self.geology_full:
                self.data[col] = self.data[col].astype('category')

        df_concat = pd.concat([self.data, self.df_train], axis=1)
        df_concat = df_concat.drop(["Sample_ID"], axis = 1)

        return df_concat


class FeatureEngineeringBin(FeatureEngineering):
    def feature_engineering_bin(self) -> pd.DataFrame:

        return pd.get_dummies(self.feature_engineering(), drop_first=True)