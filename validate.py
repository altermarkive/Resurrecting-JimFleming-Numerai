#!/usr/bin/env python

"Load data, create the validation split, train a random forest, evaluate"
"uncomment the appropriate lines to save processed data to disk"

import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import log_loss

input_file = 'data/orig/numerai_training_data.csv'

#

d = pd.read_csv( input_file )
train, val = train_test_split( d, test_size = 5000 )

# train, predict, evaluate

n_trees = 100

rf = RF( n_estimators = n_trees, verbose = True )
rf.fit( train.drop( 'target', axis = 1 ), train.target )

p = rf.predict_proba( val.drop( 'target', axis = 1 ))

ll = log_loss(val.target.values, p[:,1])
auc = AUC( val.target.values, p[:,1] )
print "AUC: {:.2%}, log loss: {:.2%}".format(auc, ll)
