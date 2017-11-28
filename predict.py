#!/usr/bin/env python

"Load data, train a random forest, output predictions"

import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RF

train_file = 'data/orig/numerai_training_data.csv'
test_file = 'data/orig/numerai_tournament_data.csv'
output_file = 'data/predictions.csv'

#

train = pd.read_csv( train_file )
test = pd.read_csv( test_file )

# train and predict

n_trees = 1000

rf = RF( n_estimators = n_trees, verbose = True )
rf.fit( train.drop( 'target', axis = 1 ), train.target )

p = rf.predict_proba( test.drop( 't_id', axis = 1 ))

# save

test['probability'] = p[:,1]
test.to_csv( output_file, columns = ( 't_id', 'probability' ), index = None )
