#!/usr/bin/env python3

"Load data, create the validation split, train a random forest, evaluate"
"uncomment the appropriate lines to save processed data to disk"

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import log_loss

import os

input_file = os.getenv('TRAINING')

#

d = pd.read_csv(input_file, header=0)
features = [f for f in list(d) if 'feature' in f]
train, val = train_test_split( d, test_size = 5000 )

# train, predict, evaluate

n_trees = 100

rf = RF( n_estimators = n_trees, verbose = True )
rf.fit(train[features], train.target)

p = rf.predict_proba(val[features])

ll = log_loss(val.target.values, p[:,1])
auc = AUC( val.target.values, p[:,1] )
print("AUC: {:.2%}, log loss: {:.2%}".format(auc, ll))
