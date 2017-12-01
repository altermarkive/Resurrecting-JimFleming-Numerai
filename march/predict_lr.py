#!/usr/bin/env python3

"Load data, scale, train a linear model, output predictions"

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.preprocessing import Normalizer, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression as LR

import os

train_file = os.getenv('TRAINING')
test_file = os.getenv('TESTING')
output_file = os.getenv('PREDICTING')

#

train = pd.read_csv(train_file, header=0)
test = pd.read_csv(test_file, header=0)
features = [f for f in list(train) if 'feature' in f]

#

y_train = train.target.values

x_train = train[features]
x_test = test[features]

transformers = [ MaxAbsScaler(), MinMaxScaler(), RobustScaler(), StandardScaler(),
        Normalizer( norm = 'l1' ), Normalizer( norm = 'l2' ), Normalizer( norm = 'max' ) ]

#poly_scaled = Pipeline([('poly', PolynomialFeatures()), ('scaler', MinMaxScaler())])
#transformers.extend([PolynomialFeatures(), poly_scaled])

selecting = int(os.getenv('SELECTING'))
if selecting != 0:
	transformer = transformers[selecting - 1]
	x_train = transformer.fit_transform(x_train)
	x_test = transformer.transform(x_test)

print("training...")

lr = LR()
lr.fit( x_train, y_train )

print("predicting...")

p = lr.predict_proba( x_test )

print("saving...")

test['probability'] = p[:,1]
test.to_csv(output_file, columns=('id', 'probability'), index=None)

# 0.69101 public
