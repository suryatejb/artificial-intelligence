# This is for INFSCI 2440 in Spring 2026.
# Task 2: Multi-category classification - predict mother's job (Mjob)
# Models: Random Forest Classifier (with SMOTE) and SVC

import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from DataLoader import prepare_task2, load_raw_data


class Task2:
    # add necessary comments to your code.
    # please feel free to create new python files, adding functions and attributes to do training, validation, testing

    def __init__(self):
        print("================Task 2================")
        train_df, test_df = load_raw_data()
        self.X_train, self.y_train, self.X_test, self.y_test = \
            prepare_task2(train_df, test_df)
        # get all unique category labels
        self.categories = sorted(set(self.y_train) | set(self.y_test))

    def print_category_results(self, category, precision, recall, f1):
        print("*" * 50)
        print("Category\t" + category + "\tF1\t" + str(f1) +
              "\tPrecision\t" + str(precision) + "\tRecall\t" + str(recall))

    def print_macro_results(self, accuracy, precision, recall, f1):
        print("*" * 50)
        print("Accuracy\t" + str(accuracy) + "\tMacro_F1\t" + str(f1) +
              "\tMacro_Precision\t" + str(precision) +
              "\tMacro_Recall\t" + str(recall))

    def _evaluate_and_print(self, y_pred):
        # pull per-category and macro metrics from sklearn's report dict
        report = classification_report(
            self.y_test, y_pred,
            labels=self.categories,
            output_dict=True,
            zero_division=0,
        )
        acc = round(accuracy_score(self.y_test, y_pred), 4)
        m_p = round(report['macro avg']['precision'], 4)
        m_r = round(report['macro avg']['recall'],    4)
        m_f = round(report['macro avg']['f1-score'],  4)
        self.print_macro_results(acc, m_p, m_r, m_f)
        for cat in self.categories:
            if cat in report:
                p = round(report[cat]['precision'], 4)
                r = round(report[cat]['recall'],    4)
                f = round(report[cat]['f1-score'],  4)
            else:
                p = r = f = 0.0
            self.print_category_results(cat, p, r, f)

    def model_1_run(self):
        print("Model 1: Random Forest Classifier (with SMOTE)")

        # SMOTE synthesises minority-class samples inside each CV fold so the
        # classifier sees a balanced distribution without leaking test data.
        # k_neighbors=2 handles the 'health' class which has very few samples.
        param_grid = {
            'clf__n_estimators':     [100, 200, 300],
            'clf__max_depth':        [None, 5, 10],
            'clf__min_samples_split':[2, 5],
            'clf__max_features':     ['sqrt', 'log2'],
            'clf__min_samples_leaf': [1, 3],
        }

        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=42, k_neighbors=2)),
            ('clf',   RandomForestClassifier(random_state=42)),
        ])

        gs = GridSearchCV(
            pipeline,
            param_grid,
            cv=10,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=0,
        )

        t0 = time.time()
        gs.fit(self.X_train, self.y_train)
        elapsed = time.time() - t0

        print("Parameters tried  : " + str(param_grid))
        print("Best parameters   : " + str(gs.best_params_))
        print("Best CV macro-F1  : " + str(round(gs.best_score_, 4)))
        print("Training time     : " + str(round(elapsed, 2)) + " s")

        y_pred = gs.best_estimator_.predict(self.X_test)
        self._evaluate_and_print(y_pred)
        return

    def model_2_run(self):
        print("--------------------\nModel 2: Support Vector Classifier (SVC)")

        # broader C range gives finer regularization control;
        # explicit small gamma values help rbf kernel with many features
        param_grid = {
            'C':      [0.01, 0.1, 1, 10, 100],
            'kernel': ['rbf', 'linear'],
            'gamma':  ['scale', 'auto', 0.001, 0.01],
        }

        gs = GridSearchCV(
            SVC(random_state=42, class_weight='balanced'),
            param_grid,
            cv=10,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=0,
        )

        t0 = time.time()
        gs.fit(self.X_train, self.y_train)
        elapsed = time.time() - t0

        print("Parameters tried  : " + str(param_grid))
        print("Best parameters   : " + str(gs.best_params_))
        print("Best CV macro-F1  : " + str(round(gs.best_score_, 4)))
        print("Training time     : " + str(round(elapsed, 2)) + " s")

        y_pred = gs.best_estimator_.predict(self.X_test)
        self._evaluate_and_print(y_pred)
        return
