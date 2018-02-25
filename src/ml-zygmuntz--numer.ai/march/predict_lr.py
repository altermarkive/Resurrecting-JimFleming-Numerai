#!/usr/bin/env python3

"Load data, scale, train a linear model, output predictions"

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.preprocessing import Normalizer, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression as LR

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

lr = LR(n_jobs=-1)
lr.fit( x_train, y_train )

print("predicting...")

p = lr.predict_proba( x_test )

print("saving...")

test['probability'] = np.clip(p[:,1], 0.0 + eps, 1.0 - eps)
test.to_csv(output_file, columns=('id', 'probability'), index=None, float_format='%.16f')

# 0.69101 public
