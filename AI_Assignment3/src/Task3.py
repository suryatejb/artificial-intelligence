# This is for INFSCI 2440 in Spring 2026.
# Task 3: Multi-label classification (tagging) - predict edusupport
# A student can have school, family, and/or paid support at the same time,
# so this is multi-label. I used OneVsRest to train one classifier per label.

import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, hamming_loss, make_scorer

from DataLoader import prepare_task3, load_raw_data


class Task3:
    # please feel free to create new python files, adding functions and attributes to do training, validation, testing

    def __init__(self):
        print("================Task 3================")
        train_df, test_df = load_raw_data()
        self.X_train, self.y_train, self.X_test, self.y_test = \
            prepare_task3(train_df, test_df)

    def model_1_run(self):
        print("Model 1: OneVsRest Logistic Regression")

        # tried different regularization strengths
        param_grid = {
            'estimator__C':        [0.01, 0.1, 1, 10, 100],
            'estimator__max_iter': [2000],
        }

        ovr = OneVsRestClassifier(
            LogisticRegression(random_state=42, solver='lbfgs')
        )
        # negate hamming loss since GridSearchCV maximizes the scoring function
        neg_hl = make_scorer(hamming_loss, greater_is_better=False)
        gs = GridSearchCV(ovr, param_grid, cv=10, scoring=neg_hl,
                          n_jobs=-1, verbose=0)

        t0 = time.time()
        gs.fit(self.X_train, self.y_train)
        elapsed = time.time() - t0

        print("Parameters tried    : " + str(param_grid))
        print("Best parameters     : " + str(gs.best_params_))
        print("Best CV Hamming loss: " + str(round(-gs.best_score_, 4)))
        print("Training time       : " + str(round(elapsed, 2)) + " s")

        y_pred = gs.best_estimator_.predict(self.X_test)
        acc = round(accuracy_score(self.y_test, y_pred), 4)
        hl  = round(hamming_loss(self.y_test,  y_pred), 4)

        print("*" * 50)
        print("Accuracy\t" + str(acc) + "\tHamming loss\t" + str(hl))
        return

    def model_2_run(self):
        print("--------------------\nModel 2: OneVsRest Gradient Boosting Classifier")

        # tried different tree counts, learning rates, and depths
        param_grid = {
            'estimator__n_estimators':  [100, 200, 300],
            'estimator__learning_rate': [0.05, 0.1, 0.2],
            'estimator__max_depth':     [3, 4, 5],
        }

        ovr = OneVsRestClassifier(
            GradientBoostingClassifier(random_state=42)
        )
        neg_hl = make_scorer(hamming_loss, greater_is_better=False)
        gs = GridSearchCV(ovr, param_grid, cv=10, scoring=neg_hl,
                          n_jobs=-1, verbose=0)

        t0 = time.time()
        gs.fit(self.X_train, self.y_train)
        elapsed = time.time() - t0

        print("Parameters tried    : " + str(param_grid))
        print("Best parameters     : " + str(gs.best_params_))
        print("Best CV Hamming loss: " + str(round(-gs.best_score_, 4)))
        print("Training time       : " + str(round(elapsed, 2)) + " s")

        y_pred = gs.best_estimator_.predict(self.X_test)
        acc = round(accuracy_score(self.y_test, y_pred), 4)
        hl  = round(hamming_loss(self.y_test,  y_pred), 4)

        print("*" * 50)
        print("Accuracy\t" + str(acc) + "\tHamming loss\t" + str(hl))
        return
