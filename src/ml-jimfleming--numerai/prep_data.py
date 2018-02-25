from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time

import random
random.seed(67)

import numpy as np
np.random.seed(67)

import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

import os

def main():
    df_train = pd.read_csv(os.getenv('TRAINING'))
    df_test = pd.read_csv(os.getenv('TESTING'))

    feature_cols = [f for f in list(df_train) if 'feature' in f]
    target_col = 'target'
    test_col = 'is_test'
    id_col = 'id'

    df_train['is_test'] = 0
    df_test['is_test'] = 1

    df_data = pd.concat([df_train, df_test])
    df_data = df_data.reindex(columns=feature_cols + [test_col, target_col])

    X_split = df_data[feature_cols]
    y_split = df_data[test_col]

    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=67)
    predictions = np.zeros(y_split.shape)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=67)
    for i, (train_i, test_i) in enumerate(kfold.split(X_split, y_split)):
        print("Fold #{}".format(i + 1))

        X_split_train = X_split.iloc[train_i]
        y_split_train = y_split.iloc[train_i]

        X_split_test = X_split.iloc[test_i]
        y_split_test = y_split.iloc[test_i]

        rf.fit(X_split_train, y_split_train)

        p = rf.predict_proba(X_split_test)[:,1]
        auc = roc_auc_score(y_split_test, p)
        print("AUC: {:.2f}".format(auc))

        predictions[test_i] = p

    # sort predictions by value
    i = predictions.argsort()

    # sort data by prediction confidence
    df_sorted = df_data.iloc[i]

    # select only training data
    df_train_sorted = df_sorted.loc[df_sorted.is_test == 0]

    # drop unnecessary columns
    df_train_sorted = df_train_sorted.drop([test_col], axis='columns')

    # verify training data
    assert(df_train_sorted[target_col].sum() == df_train[target_col].sum())

    # grab first N rows as train and last N rows as validation (those closest to test)
    validation_size = int(len(df_train_sorted) * 0.1)
    df_train = df_train_sorted.iloc[:-validation_size]
    df_valid = df_train_sorted.iloc[-validation_size:]
    print('Creating dataset with validation size: {}'.format(validation_size))

    df_train.to_csv(os.getenv('PREPARED_TRAINING'), index_label=False)
    df_valid.to_csv(os.getenv('PREPARED_VALIDATING'), index_label=False)
    df_test.to_csv(os.getenv('PREPARED_TESTING'), index_label=False)
    print('Done.')

if __name__ == '__main__':
    main()
