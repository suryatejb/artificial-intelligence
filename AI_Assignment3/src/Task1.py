# This is for INFSCI 2440 in Spring 2026.
# Task 1: Regression - predict student final grade (G3)
# Models: Gradient Boosting Regressor and Random Forest Regressor

import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

from DataLoader import prepare_task1, load_raw_data


class Task1:

    def __init__(self):
        print("================Task 1================")
        train_df, test_df = load_raw_data()
        self.X_train, self.y_train, self.X_test, self.y_test = \
            prepare_task1(train_df, test_df)

    def model_1_run(self):
        print("Model 1: Gradient Boosting Regressor")

        # Builds trees sequentially, each one correcting residuals from the last.
        # subsample < 1.0 adds stochasticity to reduce overfitting.
        # A very small learning_rate with more estimators generalizes better.
        param_grid = {
            'n_estimators':    [100, 200, 300],
            'learning_rate':   [0.01, 0.05, 0.1],
            'max_depth':       [3, 4, 5],
            'subsample':       [0.8, 1.0],
            'min_samples_leaf':[1, 5],
        }

        gs = GridSearchCV(
            GradientBoostingRegressor(random_state=42),
            param_grid,
            cv=10,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0,
        )

        t0 = time.time()
        gs.fit(self.X_train, self.y_train)
        elapsed = time.time() - t0

        print("Parameters tried : " + str(param_grid))
        print("Best parameters  : " + str(gs.best_params_))
        print("Best CV MSE      : " + str(round(-gs.best_score_, 4)))
        print("Training time    : " + str(round(elapsed, 2)) + " s")

        # test with the best model found
        y_pred   = gs.best_estimator_.predict(self.X_test)
        test_mse = mean_squared_error(self.y_test, y_pred)

        print("*" * 50)
        print("Mean squared error      " + str(round(test_mse, 4)))
        return

    def model_2_run(self):
        print("--------------------\nModel 2: Random Forest Regressor")

        # max_features limits features per split -> decorrelates trees
        # min_samples_leaf regularizes leaves to reduce overfitting
        param_grid = {
            'n_estimators':     [100, 200, 300],
            'max_depth':        [None, 5, 10],
            'min_samples_split':[2, 5],
            'min_samples_leaf': [1, 5],
            'max_features':     ['sqrt', 0.7],
        }

        gs = GridSearchCV(
            RandomForestRegressor(random_state=42),
            param_grid,
            cv=10,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0,
        )

        t0 = time.time()
        gs.fit(self.X_train, self.y_train)
        elapsed = time.time() - t0

        print("Parameters tried : " + str(param_grid))
        print("Best parameters  : " + str(gs.best_params_))
        print("Best CV MSE      : " + str(round(-gs.best_score_, 4)))
        print("Training time    : " + str(round(elapsed, 2)) + " s")

        y_pred   = gs.best_estimator_.predict(self.X_test)
        test_mse = mean_squared_error(self.y_test, y_pred)

        print("*" * 50)
        print("Mean squared error\t" + str(round(test_mse, 4)))
        return
