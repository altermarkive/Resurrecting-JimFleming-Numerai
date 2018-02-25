from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
import time
import random
random.seed(67)

import numpy as np
np.random.seed(67)

import pandas as pd

from sklearn.decomposition import RandomizedPCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline, make_union
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.manifold import Isomap

from transformers import ItemSelector

import os

def main():
    # load data
    df_train = pd.read_csv(os.getenv('PREPARED_TRAINING'))
    df_valid = pd.read_csv(os.getenv('PREPARED_VALIDATING'))
    df_test = pd.read_csv(os.getenv('PREPARED_TESTING'))

    feature_cols = list(df_train.columns[:-1])
    target_col = df_train.columns[-1]

    X_train = df_train[feature_cols].values
    y_train = df_train[target_col].values

    X_valid = df_valid[feature_cols].values
    y_valid = df_valid[target_col].values

    X_test = df_test[feature_cols].values

    prefix = os.getenv('STORING')
    tsne_data_2d_5p = np.load(os.path.join(prefix, 'tsne_2d_5p.npz'))
    tsne_data_2d_10p = np.load(os.path.join(prefix, 'tsne_2d_10p.npz'))
    tsne_data_2d_15p = np.load(os.path.join(prefix, 'tsne_2d_15p.npz'))
    tsne_data_2d_30p = np.load(os.path.join(prefix, 'tsne_2d_30p.npz'))
    tsne_data_2d_50p = np.load(os.path.join(prefix, 'tsne_2d_50p.npz'))
    tsne_2d_only = bool(int(os.getenv('TSNE_2D_ONLY', '0')))
    if not tsne_2d_only:
        tsne_data_3d_30p = np.load(os.path.join(prefix, 'tsne_3d_30p.npz'))

    # concat features
    X_train_concat = {
        'X': X_train,
        'tsne_2d_5p': tsne_data_2d_5p['train'],
        'tsne_2d_10p': tsne_data_2d_10p['train'],
        'tsne_2d_15p': tsne_data_2d_15p['train'],
        'tsne_2d_30p': tsne_data_2d_30p['train'],
        'tsne_2d_50p': tsne_data_2d_50p['train'],
    }
    if not tsne_2d_only:
        X_train_concat['tsne_3d_30p'] = tsne_data_3d_30p['train']
    X_valid_concat = {
        'X': X_valid,
        'tsne_2d_5p': tsne_data_2d_5p['valid'],
        'tsne_2d_10p': tsne_data_2d_10p['valid'],
        'tsne_2d_15p': tsne_data_2d_15p['valid'],
        'tsne_2d_30p': tsne_data_2d_30p['valid'],
        'tsne_2d_50p': tsne_data_2d_50p['valid'],
    }
    if not tsne_2d_only:
        X_valid_concat['tsne_3d_30p'] = tsne_data_3d_30p['valid']
    X_test_concat = {
        'X': X_test,
        'tsne_2d_5p': tsne_data_2d_5p['test'],
        'tsne_2d_10p': tsne_data_2d_10p['test'],
        'tsne_2d_15p': tsne_data_2d_15p['test'],
        'tsne_2d_30p': tsne_data_2d_30p['test'],
        'tsne_2d_50p': tsne_data_2d_50p['test'],
    }
    if not tsne_2d_only:
        X_test_concat['tsne_3d_30p'] = tsne_data_3d_30p['test']

    # build pipeline
    transformers = [
        ('X', ItemSelector('X')),
        ('tsne_2d_5p', ItemSelector('tsne_2d_5p')),
        ('tsne_2d_10p', ItemSelector('tsne_2d_10p')),
        ('tsne_2d_15p', ItemSelector('tsne_2d_15p')),
        ('tsne_2d_30p', ItemSelector('tsne_2d_30p')),
        ('tsne_2d_50p', ItemSelector('tsne_2d_50p')),
    ]
    if not tsne_2d_only:
        transformers.append(('tsne_3d_30p', ItemSelector('tsne_3d_30p')))
    classifier = Pipeline(steps=[
        ('features', FeatureUnion(transformer_list=transformers)),
        ('poly', PolynomialFeatures(degree=2)),
        ('scaler', StandardScaler()),
        ('gbt', GradientBoostingClassifier()),
    ])

    print('Fitting...')
    start_time = time.time()
    classifier.fit(X_train_concat, y_train)
    print('Fit: {}s'.format(time.time() - start_time))

    p_valid = classifier.predict_proba(X_valid_concat)
    loss = log_loss(y_valid, p_valid)
    print('Loss: {}'.format(loss))

    p_test = classifier.predict_proba(X_test_concat)
    df_pred = pd.DataFrame({
        'id': df_test['id'],
        'probability': p_test[:,1]
    })
    csv_path = os.getenv('PREDICTING')
    df_pred.to_csv(csv_path, columns=('id', 'probability'), index=None)
    print('Saved: {}'.format(csv_path))

if __name__ == '__main__':
    main()
