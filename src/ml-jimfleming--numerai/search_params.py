from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import pandas as pd
import sklearn.pipeline as pipeline
import sklearn.decomposition as decomposition
import sklearn.linear_model as linear_model
import sklearn.model_selection as model_selection
import sklearn.preprocessing as preprocessing

import os

def build_pipeline(portion):
    return pipeline.Pipeline([('poly', preprocessing.PolynomialFeatures(degree=2)), ('pca', decomposition.PCA()), ('logisticregression', linear_model.LogisticRegression())])

def main():
    df_train = pd.read_csv(os.getenv('PREPARED_TRAINING'))
    df_valid = pd.read_csv(os.getenv('PREPARED_VALIDATING'))

    feature_cols = list(df_train.columns[:-1])
    target_col = df_train.columns[-1]

    X_train = df_train[feature_cols].values
    y_train = df_train[target_col].values

    X_valid = df_valid[feature_cols].values
    y_valid = df_valid[target_col].values

    tsne_data = np.load(os.path.join(os.getenv('STORING'), 'tsne.npz'))
    X_train_tsne = tsne_data['X_train']
    X_valid_tsne = tsne_data['X_valid']

    params = {
        # 'featureunion__polynomialfeatures__degree': range(2, 4),
        # 'featureunion__portionkernelpca__n_components': range(2, 102, 2),
        # 'featureunion__portionkernelpca__degree': range(2, 4),
        # 'featureunion__portionkernelpca__kernel': ['cosine', 'rbf'],
        # 'featureunion__portionisomap__n_neighbors': range(1, 11),
        # 'featureunion__portionisomap__n_components': range(2, 102, 2),
        'pca__n_components': list(range(2, 202, 2)),
        'pca__whiten': [True, False],
        'logisticregression__C': [1e-4, 1e-3, 1e-2, 1e-1, 1e-0],
        'logisticregression__penalty': ['l1', 'l2']
    }

    pipeline = build_pipeline(portion=0.1)

    X_search = np.concatenate([
            np.concatenate([X_train, X_train_tsne], axis=1),
            np.concatenate([X_valid, X_valid_tsne], axis=1),
            ], axis=0)
    y_search = np.concatenate([y_train, y_valid], axis=0)

    train_indices = range(0, len(X_train))
    valid_indices = range(len(X_train), len(X_train)+len(X_valid))
    assert(len(train_indices) == len(X_train))
    assert(len(valid_indices) == len(X_valid))

    cv = [(train_indices, valid_indices)]

    search = model_selection.RandomizedSearchCV(pipeline, params, cv=cv, n_iter=100, verbose=2)
    search.fit(X_search, y_search)

    print(search.best_score_)
    print(search.best_params_)

if __name__ == '__main__':
    main()
