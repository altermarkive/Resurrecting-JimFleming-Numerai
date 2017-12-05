#!/usr/bin/env python3

"Load data, train a random forest, output predictions"

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RF

import os
import sys

eps = sys.float_info.epsilon

train_file = os.getenv('TRAINING')
test_file = os.getenv('TESTING')
output_file = os.getenv('PREDICTING')

#

train = pd.read_csv(train_file, header=0)
test = pd.read_csv(test_file, header=0)
features = [f for f in list(train) if 'feature' in f]

# train and predict

n_trees = 1000

rf = RF(n_estimators = n_trees, verbose = True, n_jobs=-1)
rf.fit(train[features], train.target)

p = rf.predict_proba(test[features])

# save

test['probability'] = np.clip(p[:,1], 0.0 + eps, 1.0 - eps)
test.to_csv(output_file, columns=('id', 'probability'), index=None, float_format='%.16f')
