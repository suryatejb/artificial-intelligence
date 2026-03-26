# DataLoader.py
# shared helper for loading and encoding the dataset across all 3 tasks
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# column names in the order they appear in the file (attributes 1-28)
COLUMNS = [
    'school', 'sex', 'age', 'address', 'famsize', 'Pstatus',
    'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian',
    'traveltime', 'studytime', 'failures', 'edusupport',
    'nursery', 'higher', 'internet', 'romantic',
    'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G3'
]

# build file paths relative to this script so it works regardless of cwd
_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.dirname(_SRC_DIR)
TRAIN_PATH = os.path.join(_DATA_DIR, 'assign3_students_train.txt')
TEST_PATH  = os.path.join(_DATA_DIR, 'assign3_students_test.txt')

# binary string columns - just map them to 0/1
BINARY_MAPS = {
    'school':  {'GP': 0, 'MS': 1},
    'sex':     {'F': 0, 'M': 1},
    'address': {'U': 0, 'R': 1},
    'famsize': {'LE3': 0, 'GT3': 1},
    'Pstatus': {'T': 0, 'A': 1},
    'nursery': {'no': 0, 'yes': 1},
    'higher':  {'no': 0, 'yes': 1},
    'internet':{'no': 0, 'yes': 1},
    'romantic':{'no': 0, 'yes': 1},
}

# these have more than 2 categories and no ordinal meaning, so one-hot encode
NOMINAL_COLS = ['Mjob', 'Fjob', 'reason', 'guardian']

# a student can have any combination of these support types
EDU_LABELS = ['school', 'family', 'paid']


def load_raw_data():
    train_df = pd.read_csv(TRAIN_PATH, sep='\t', header=None, names=COLUMNS)
    test_df  = pd.read_csv(TEST_PATH,  sep='\t', header=None, names=COLUMNS)
    return train_df, test_df


def _binarize_edusupport(series):
    # edusupport can be "school family", "paid", "no", etc.
    # build a binary matrix with one column per support type
    Y = np.zeros((len(series), len(EDU_LABELS)), dtype=int)
    for i, val in enumerate(series):
        tokens = str(val).split()
        for j, lbl in enumerate(EDU_LABELS):
            if lbl in tokens:
                Y[i, j] = 1
    return Y


def _add_edu_binary_features(df):
    # when edusupport is an input feature, split it into 3 separate binary cols
    df = df.copy()
    for lbl in EDU_LABELS:
        df[f'edu_{lbl}'] = df['edusupport'].apply(
            lambda x: 1 if lbl in str(x).split() else 0
        )
    df = df.drop(columns=['edusupport'])
    return df


def _encode(combined_df, target_col):
    df = combined_df.copy()

    # map binary string columns to 0/1
    for col, mapping in BINARY_MAPS.items():
        if col != target_col and col in df.columns:
            df[col] = df[col].map(mapping)

    # expand edusupport into 3 binary features (skip when it's the target)
    if target_col != 'edusupport' and 'edusupport' in df.columns:
        df = _add_edu_binary_features(df)

    # one-hot encode nominal columns
    ohe_cols = [c for c in NOMINAL_COLS if c != target_col and c in df.columns]
    if ohe_cols:
        df = pd.get_dummies(df, columns=ohe_cols, drop_first=False, dtype=float)

    # drop the target so it's not used as a feature
    if target_col in df.columns:
        df = df.drop(columns=[target_col])

    return df


def prepare_task1(train_df, test_df):
    # Task 1: predict G3, use all other columns as features
    target  = 'G3'
    y_train = train_df[target].values.astype(float)
    y_test  = test_df[target].values.astype(float)

    n_train  = len(train_df)
    combined = pd.concat([train_df, test_df], ignore_index=True)
    encoded  = _encode(combined, target_col=target)

    X       = encoded.values.astype(float)
    X_train = X[:n_train]
    X_test  = X[n_train:]

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test


def prepare_task2(train_df, test_df):
    # Task 2: predict Mjob, exclude it from the feature set
    target  = 'Mjob'
    y_train = train_df[target].values
    y_test  = test_df[target].values

    n_train  = len(train_df)
    combined = pd.concat([train_df, test_df], ignore_index=True)
    encoded  = _encode(combined, target_col=target)

    X       = encoded.values.astype(float)
    X_train = X[:n_train]
    X_test  = X[n_train:]

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test


def prepare_task3(train_df, test_df):
    # Task 3: predict edusupport as multi-label
    # labels are [school, family, paid] - each student can have any subset
    target  = 'edusupport'
    y_train = _binarize_edusupport(train_df[target].values)
    y_test  = _binarize_edusupport(test_df[target].values)

    n_train  = len(train_df)
    combined = pd.concat([train_df, test_df], ignore_index=True)
    encoded  = _encode(combined, target_col=target)

    X       = encoded.values.astype(float)
    X_train = X[:n_train]
    X_test  = X[n_train:]

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test
