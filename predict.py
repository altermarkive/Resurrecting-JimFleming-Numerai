#!/usr/bin/env python3

"Load data, train a random forest, output predictions"

import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RF

import os

train_file = os.getenv('TRAINING')
test_file = os.getenv('TESTING')
output_file = os.getenv('PREDICTING')

#

train = pd.read_csv(train_file, header=0)
test = pd.read_csv(test_file, header=0)
features = [f for f in list(train) if 'feature' in f]

# train and predict

n_trees = 1000

rf = RF( n_estimators = n_trees, verbose = True )
rf.fit(train[features], train.target)

p = rf.predict_proba(test[features])

# save

test['probability'] = p[:,1]
test.to_csv(output_file, columns=('id', 'probability'), index=None)
