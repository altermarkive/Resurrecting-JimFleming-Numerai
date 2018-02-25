#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import numpy as np
import pandas as pd
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.preprocessing import PolynomialFeatures


def save_tsne(perplexity, polynomial):
    prefix = os.getenv('STORING')
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

    X_all = np.concatenate([X_train, X_valid, X_test], axis=0)

    if polynomial:
        poly = PolynomialFeatures(degree=2)
        X_all = poly.fit_transform(X_all)

    template = 'Running 2D TSNE; perplexity: {}, polynomial: {}\n'
    sys.stdout.write(template.format(perplexity, polynomial))
    sys.stdout.flush()
    start_time = time.time()
    count = int(os.environ.get('PARALLEL', '-1'))
    tsne_all = TSNE(
        n_components=2, perplexity=perplexity, n_jobs=count
    ).fit_transform(X_all)
    sys.stdout.write('TSNE completed: {}s\n'.format(time.time() - start_time))
    sys.stdout.flush()

    end = X_train.shape[0]
    tsne_train = tsne_all[:end]
    assert(len(tsne_train) == len(X_train))

    begin = X_train.shape[0]
    end = X_train.shape[0] + X_valid.shape[0]
    tsne_valid = tsne_all[begin:end]
    assert(len(tsne_valid) == len(X_valid))

    begin = X_train.shape[0] + X_valid.shape[0]
    end = X_train.shape[0] + X_valid.shape[0] + X_test.shape[0]
    tsne_test = tsne_all[begin:end]
    assert(len(tsne_test) == len(X_test))

    if polynomial:
        save_path = 'tsne_2d_{}p_poly.npz'.format(perplexity)
    else:
        save_path = 'tsne_2d_{}p.npz'.format(perplexity)
    save_path = os.path.join(prefix, save_path)

    np.savez(save_path, train=tsne_train, valid=tsne_valid, test=tsne_test)
    sys.stdout.write('Saved: {}\n'.format(save_path))
    sys.stdout.flush()


def main():
    perplexity = os.getenv('TSNE_PERPLEXITY', '5,10,15,30,50,5,10,15,30,50')
    perplexity = [int(value) for value in perplexity.split(',')]
    polynomial = os.getenv('TSNE_POLYNOMIAL', '0,0,0,0,0,1,1,1,1,1')
    polynomial = [bool(int(value)) for value in polynomial.split(',')]
    for i in range(min(len(perplexity), len(polynomial))):
        save_tsne(perplexity[i], polynomial[i])


if __name__ == '__main__':
    main()
