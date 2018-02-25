from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import random
random.seed(67)

import numpy as np
np.random.seed(67)

import pandas as pd

from tpot import TPOTClassifier

import os

def main():
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
    tsne_data = np.load(os.path.join(prefix, 'tsne_2d_5p.npz'))
    tsne_train = tsne_data['train']
    tsne_valid = tsne_data['valid']
    tsne_test = tsne_data['test']

    # concat features
    X_train_concat = np.concatenate([X_train, tsne_train], axis=1)
    X_valid_concat = np.concatenate([X_valid, tsne_valid], axis=1)
    X_test_concat = np.concatenate([X_test, tsne_test], axis=1)

    tpot = TPOTClassifier(
        max_time_mins=int(os.getenv('TIME_LIMIT_ALL', '1440')),
        max_eval_time_mins=int(os.getenv('TIME_LIMIT_PART', '5')),
        population_size=100,
        scoring='log_loss',
        cv=3,
        verbosity=2,
        random_state=67)
    tpot.fit(X_train_concat, y_train)
    loss = tpot.score(X_valid_concat, y_valid)
    print(loss)
    tpot.export(os.path.join(prefix, 'tpot_pipeline.py'))

    p_test = tpot.predict_proba(X_test_concat)
    df_pred = pd.DataFrame({
        'id': df_test['id'],
        'probability': p_test[:,1]
    })
    csv_path = os.getenv('PREDICTING')
    df_pred.to_csv(csv_path, columns=('id', 'probability'), index=None)
    print('Saved: {}'.format(csv_path))

if __name__ == '__main__':
    main()
