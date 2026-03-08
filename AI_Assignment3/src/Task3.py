# This is for INFSCI 2440 in Spring 2026.
# Task 3: Multi-label classification task (tagging)
#
# Predict edusupport, which can be ANY COMBINATION of {school, family, paid}
# or "no" (none).  Each token is treated as an independent binary label so
# label vectors like [1, 1, 0] ("school family") are valid.
#
#   Model 1 - One-vs-Rest  Logistic Regression
#   Model 2 - One-vs-Rest  Gradient Boosting Classifier
#
# Hyperparameter tuning: 10-fold cross-validation (GridSearchCV).
# Evaluation: subset Accuracy and Hamming Loss.
#
# Feature engineering (edusupport excluded as a feature):
#   - Binary categoricals (school, sex, …) → 0/1
#   - Nominal columns (Mjob, Fjob, reason, guardian) → one-hot encoding
#   - All features standardised with StandardScaler

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

    # ------------------------------------------------------------------ #
    # Model 1 – One-vs-Rest  Logistic Regression
    # ------------------------------------------------------------------ #
    def model_1_run(self):
        print("Model 1: OneVsRest Logistic Regression")

        # Parameters explored during 10-fold CV tuning
        param_grid = {
            'estimator__C':        [0.01, 0.1, 1, 10, 100],
            'estimator__max_iter': [2000],
        }

        ovr = OneVsRestClassifier(
            LogisticRegression(random_state=42, solver='lbfgs')
        )
        # Use negative hamming loss so GridSearchCV maximises it (minimises HL)
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

    # ------------------------------------------------------------------ #
    # Model 2 – One-vs-Rest  Gradient Boosting Classifier
    # ------------------------------------------------------------------ #
    def model_2_run(self):
        print("--------------------\nModel 2: OneVsRest Gradient Boosting Classifier")

        # Parameters explored during 10-fold CV tuning
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
