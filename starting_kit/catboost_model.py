import pandas as pd
import math

import numpy as np
from catboost import Pool, CatBoostClassifier, CatBoostRegressor


class RmseObjective(object):
    def calc_ders_range(self, approxes, targets, weights):
        assert len(approxes) == len(targets)
        if weights is not None:
            assert len(weights) == len(approxes)

        result = []
        for index in range(len(targets)):
            der1 = targets[index] - approxes[index]
            der2 = -1

            if weights is not None:
                der1 *= weights[index]
                der2 *= weights[index]

            result.append((der1, der2))
        return result


class Catboost_Model:
    def __init__(self, train_data, level_col, test, features,
                 cat_features, targets, learning_rate=0.01, iterations=300, depth=5, classif = False):
        self.train_data = train_data
        self.model = None
        self.level_col = level_col
        self.test = test
        self.features = features
        self.cat_features = cat_features
        self.targets = targets
        self.learning_rate = learning_rate
        self.classif = classif
        self.depth = depth
        self.iterations = iterations

    def train(self):
        """
        Train a Catboost model

        inputs:
            train_data (pandas DataFrame): Training DataFrame
            test (pandas DataFrame): Testing DataFrame
            features (list[str]): List of columns to consider as variables during the training
            cat_features (list[int]): List of categorical labels
            targets (list[str]): Targets
            learning_rate (float): Learning rate
            iterations (int): Number of iterations
            depth (int): Depth of model
            classif (bool): If true, use a classification model, use regression otherwise

        output:
            model (CatBoost): CatBoost fit model

        """
        models = []
        for i, target in enumerate(self.targets):
            print("==== Target ", target, " ====")
            print("Iteration ", i + 1, "/", len(self.targets))

            relevant = self.train_data[self.features + [target]].dropna()

            last_trend = relevant['trend'].max()

            weights = relevant.trend.map(lambda t: math.exp((t - last_trend) / 96 / 30))
            relevant = relevant[self.features + [target]]

            valid = self.test[self.features + [target]].dropna()

            # Training model

            train_dataset = Pool(data=relevant[self.features],
                                 label=relevant[target],
                                 cat_features=self.cat_features,
                                 weight=weights)
            valid_dataset = Pool(data=valid[self.features],
                                 label=valid[target],
                                 cat_features=self.cat_features)
            if self.classif:
                clf = CatBoostClassifier(iterations=self.iterations,
                                         learning_rate=self.learning_rate,
                                         depth=self.depth,
                                         loss_function="MultiClass")
                clf.fit(train_dataset, eval_set=valid_dataset)
            else:
                clf = CatBoostRegressor(iterations=self.iterations,
                                        learning_rate=self.learning_rate,
                                        depth=self.depth,
                                        loss_function=RmseObjective(),
                                        eval_metric="MAE")
                clf.fit(train_dataset, eval_set=valid_dataset)  # , sample_weight=weights)
            models.append(clf)
            self.model = models

    def catboost_test_score(self):
        """
        Test score of the Catboost model

        input:
            model (CatBoost): CatBoost model

        output:
            (float): Score
        """
        return round(self.model.get_best_score()['validation']['MAE'], 3)

    def predict(self, test):
        """
        Prediction using CatBoost model

        inputs:
            models (list[CatBoost]): List of catboost models
            test (pandas DataFrame): Testing DataFrame
            features (list[str]): List of columns to consider as variables during the training
            targets (list[str]): Targets
            level_col ():
        """

        relevant = test[self.features + ['date']
                        ].dropna().reset_index(drop=True)

        for i, _ in enumerate(self.model):
            print("==== Target ", self.targets[i], " ====")
            print("Iteration ", i + 1, "/", len(self.model))

            # Getting Predictions

            relevant[self.targets[i]] = self.model[i].predict(relevant).round()

        if self.level_col == 'Station':
            relevant = relevant.merge(test[['Station', 'area']].value_counts(
            ).reset_index().drop(0, axis=1), on=['Station', 'area'], how='left')
        elif self.level_col == 'area':
            relevant = relevant.merge(test[['area']].value_counts(
            ).reset_index().drop(0, axis=1), on=['area'], how='left')
        return relevant

    def mae(y_true, y_pred, N):
        """
        Mean absolute error

        inputs:
            y_true (NumpyArray): ground truth
            y_pred (NumpyArray): prediction
            N (int): Number of samples
        """
        return round(sum(abs(y_true - y_pred)) / N, 1)

    def sae(y_true, y_pred):
        """
        Sum of absolute errors

        inputs:
            y_true (NumpyArray): ground truth
            y_pred (NumpyArray): prediction
        """
        return round(np.sum(abs(y_true - y_pred)), 3)
