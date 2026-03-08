"""
DataLoader.py
Utility module for loading and preprocessing the Student Portuguese Class
Performance dataset for Assignment 3.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Column definitions
# --------------------------------------------------------------------------- #
COLUMNS = [
    'school', 'sex', 'age', 'address', 'famsize', 'Pstatus',
    'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian',
    'traveltime', 'studytime', 'failures', 'edusupport',
    'nursery', 'higher', 'internet', 'romantic',
    'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G3'
]

# Resolve paths relative to this file so the code works regardless of cwd
_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.dirname(_SRC_DIR)
TRAIN_PATH = os.path.join(_DATA_DIR, 'assign3_students_train.txt')
TEST_PATH  = os.path.join(_DATA_DIR, 'assign3_students_test.txt')

# --------------------------------------------------------------------------- #
# Encoding constants
# --------------------------------------------------------------------------- #
# True binary columns (two distinct string values)
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

# Multi-valued nominal columns (will be one-hot encoded)
NOMINAL_COLS = ['Mjob', 'Fjob', 'reason', 'guardian']

# Multi-label edusupport tokens
EDU_LABELS = ['school', 'family', 'paid']

# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_raw_data():
    """Return (train_df, test_df) as pandas DataFrames with named columns."""
    train_df = pd.read_csv(TRAIN_PATH, sep='\t', header=None, names=COLUMNS)
    test_df  = pd.read_csv(TEST_PATH,  sep='\t', header=None, names=COLUMNS)
    return train_df, test_df


# --------------------------------------------------------------------------- #
# edusupport helpers
# --------------------------------------------------------------------------- #
def _binarize_edusupport(series):
    """
    Convert the edusupport Series (which may contain space-separated tokens
    such as 'school family' or a single 'no') to a binary NumPy matrix with
    columns [school_support, family_support, paid_support].
    """
    Y = np.zeros((len(series), len(EDU_LABELS)), dtype=int)
    for i, val in enumerate(series):
        tokens = str(val).split()
        for j, lbl in enumerate(EDU_LABELS):
            if lbl in tokens:
                Y[i, j] = 1
    return Y


def _add_edu_binary_features(df):
    """
    Replace the edusupport column with three binary feature columns
    (edu_school, edu_family, edu_paid) for use as *input features*.
    """
    df = df.copy()
    for lbl in EDU_LABELS:
        df[f'edu_{lbl}'] = df['edusupport'].apply(
            lambda x: 1 if lbl in str(x).split() else 0
        )
    df = df.drop(columns=['edusupport'])
    return df


# --------------------------------------------------------------------------- #
# Core encoding pipeline
# --------------------------------------------------------------------------- #
def _encode(combined_df, target_col):
    """
    Encode all feature columns of a combined (train + test) DataFrame:
      - Binary string columns  -> 0 / 1
      - edusupport (if not target) -> 3 binary columns (edu_school, …)
      - Nominal columns (if not target) -> one-hot via pd.get_dummies
      - Drop target column

    Returns the encoded DataFrame (float dtype ready for sklearn).
    """
    df = combined_df.copy()

    # 1. Binary encode
    for col, mapping in BINARY_MAPS.items():
        if col != target_col and col in df.columns:
            df[col] = df[col].map(mapping)

    # 2. edusupport: multi-label binary features (unless it is the target)
    if target_col != 'edusupport' and 'edusupport' in df.columns:
        df = _add_edu_binary_features(df)

    # 3. One-hot encode nominal columns (skip if it's the target)
    ohe_cols = [c for c in NOMINAL_COLS if c != target_col and c in df.columns]
    if ohe_cols:
        df = pd.get_dummies(df, columns=ohe_cols, drop_first=False, dtype=float)

    # 4. Drop the target column
    if target_col in df.columns:
        df = df.drop(columns=[target_col])

    return df


# --------------------------------------------------------------------------- #
# Task-specific preparation
# --------------------------------------------------------------------------- #
def prepare_task1(train_df, test_df):
    """
    Task 1 – Predict G3 (final grade, regression).
    Features: all 27 other attributes (edusupport -> 3 binary cols; nominals -> OHE).
    Returns: X_train, y_train, X_test, y_test  (X scaled via StandardScaler)
    """
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
    """
    Task 2 – Predict Mjob (mother's job, multi-class classification).
    Features: all attributes except Mjob.
    Returns: X_train, y_train, X_test, y_test  (X scaled via StandardScaler)
    """
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
    """
    Task 3 – Predict edusupport (multi-label classification / tagging).
    Labels: [school_support, family_support, paid_support]
            (a student may have any subset; 'no'  -> all zeros)
    Features: all attributes except edusupport.
    Returns: X_train, y_train (binary matrix), X_test, y_test  (X scaled)
    """
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
