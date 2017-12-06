#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 19:25:57 2016

@author: jeremy
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
import itertools
import pandas as pd
import tensorflow as tf

flags = tf.app.flags

COLUMNS=[
'feature1',
'feature2',
'feature3',
'feature4',
'feature5',
'feature6',
'feature7',
'feature8',
'feature9',
'feature10',
'feature11',
'feature12',
'feature13',
'feature14',
'feature15',
'feature16',
'feature17',
'feature18',
'feature19',
'feature20',
'feature21',
'feature22',
'feature23',
'feature24',
'feature25',
'feature26',
'feature27',
'feature28',
'feature29',
'feature30',
'feature31',
'feature32',
'feature33',
'feature34',
'feature35',
'feature36',
'feature37',
'feature38',
'feature39',
'feature40',
'feature41',
'feature42',
'feature43',
'feature44',
'feature45',
'feature46',
'feature47',
'feature48',
'feature49',
'feature50',]


CONTINUOUS_COLUMNS = [
'feature1',
'feature2',
'feature3',
'feature4',
'feature5',
'feature6',
'feature7',
'feature8',
'feature9',
'feature10',
'feature11',
'feature12',
'feature13',
'feature14',
'feature15',
'feature16',
'feature17',
'feature18',
'feature19',
'feature20',
'feature21',
'feature22',
'feature23',
'feature24',
'feature25',
'feature26',
'feature27',
'feature28',
'feature29',
'feature30',
'feature31',
'feature32',
'feature33',
'feature34',
'feature35',
'feature36',
'feature37',
'feature38',
'feature39',
'feature40',
'feature41',
'feature42',
'feature43',
'feature44',
'feature45',
'feature46',
'feature47',
'feature48',
'feature49',
'feature50',]


def build_estimator(model_dir,model_type):
  """Build an estimator."""
  # Continuous base columns.
  feature1 = tf.contrib.layers.real_valued_column('feature1')
  feature2 = tf.contrib.layers.real_valued_column('feature2')
  feature3 = tf.contrib.layers.real_valued_column('feature3')
  feature4 = tf.contrib.layers.real_valued_column('feature4')
  feature5 = tf.contrib.layers.real_valued_column('feature5')
  feature6 = tf.contrib.layers.real_valued_column('feature6')
  feature7 = tf.contrib.layers.real_valued_column('feature7')
  feature8 = tf.contrib.layers.real_valued_column('feature8')
  feature9 = tf.contrib.layers.real_valued_column('feature9')
  feature10 = tf.contrib.layers.real_valued_column('feature10')
  feature11 = tf.contrib.layers.real_valued_column('feature11')
  feature12 = tf.contrib.layers.real_valued_column('feature12')
  feature13 = tf.contrib.layers.real_valued_column('feature13')
  feature14 = tf.contrib.layers.real_valued_column('feature14')
  feature15 = tf.contrib.layers.real_valued_column('feature15')
  feature16 = tf.contrib.layers.real_valued_column('feature16')
  feature17 = tf.contrib.layers.real_valued_column('feature17')
  feature18 = tf.contrib.layers.real_valued_column('feature18')
  feature19 = tf.contrib.layers.real_valued_column('feature19')
  feature20 = tf.contrib.layers.real_valued_column('feature20')
  feature21 = tf.contrib.layers.real_valued_column('feature21')
  feature22 = tf.contrib.layers.real_valued_column('feature22')
  feature23 = tf.contrib.layers.real_valued_column('feature23')
  feature24 = tf.contrib.layers.real_valued_column('feature24')
  feature25 = tf.contrib.layers.real_valued_column('feature25')
  feature26 = tf.contrib.layers.real_valued_column('feature26')
  feature27 = tf.contrib.layers.real_valued_column('feature27')
  feature28 = tf.contrib.layers.real_valued_column('feature28')
  feature29 = tf.contrib.layers.real_valued_column('feature29')
  feature30 = tf.contrib.layers.real_valued_column('feature30')
  feature31 = tf.contrib.layers.real_valued_column('feature31')
  feature32 = tf.contrib.layers.real_valued_column('feature32')
  feature33 = tf.contrib.layers.real_valued_column('feature33')
  feature34 = tf.contrib.layers.real_valued_column('feature34')
  feature35 = tf.contrib.layers.real_valued_column('feature35')
  feature36 = tf.contrib.layers.real_valued_column('feature36')
  feature37 = tf.contrib.layers.real_valued_column('feature37')
  feature38 = tf.contrib.layers.real_valued_column('feature38')
  feature39 = tf.contrib.layers.real_valued_column('feature39')
  feature40 = tf.contrib.layers.real_valued_column('feature40')
  feature41 = tf.contrib.layers.real_valued_column('feature41')
  feature42 = tf.contrib.layers.real_valued_column('feature42')
  feature43 = tf.contrib.layers.real_valued_column('feature43')
  feature44 = tf.contrib.layers.real_valued_column('feature44')
  feature45 = tf.contrib.layers.real_valued_column('feature45')
  feature46 = tf.contrib.layers.real_valued_column('feature46')
  feature47 = tf.contrib.layers.real_valued_column('feature47')
  feature48 = tf.contrib.layers.real_valued_column('feature48')
  feature49 = tf.contrib.layers.real_valued_column('feature49')
  feature50 = tf.contrib.layers.real_valued_column('feature50')

  
  # Wide columns and deep columns.
  wide_columns = [
        feature1,
        feature2,
        feature3,
        feature4,
        feature5,
        feature6,
        feature7,
        feature8,
        feature9,
        feature10,
        feature11,
        feature12,
        feature13,
        feature14,
        feature15,
        feature16,
        feature17,
        feature18,
        feature19,
        feature20,
        feature21,
        feature22,
        feature23,
        feature24,
        feature25,
        feature26,
        feature27,
        feature28,
        feature29,
        feature30,
        feature31,
        feature32,
        feature33,
        feature34,
        feature35,
        feature36,
        feature37,
        feature38,
        feature39,
        feature40,
        feature41,
        feature42,
        feature43,
        feature44,
        feature45,
        feature46,
        feature47,
        feature48,
        feature49,
        feature50,]
  deep_columns = [
        feature1,
        feature2,
        feature3,
        feature4,
        feature5,
        feature6,
        feature7,
        feature8,
        feature9,
        feature10,
        feature11,
        feature12,
        feature13,
        feature14,
        feature15,
        feature16,
        feature17,
        feature18,
        feature19,
        feature20,
        feature21,
        feature22,
        feature23,
        feature24,
        feature25,
        feature26,
        feature27,
        feature28,
        feature29,
        feature30,
        feature31,
        feature32,
        feature33,
        feature34,
        feature35,
        feature36,
        feature37,
        feature38,
        feature39,
        feature40,
        feature41,
        feature42,
        feature43,
        feature44,
        feature45,
        feature46,
        feature47,
        feature48,
        feature49,
        feature50,]
  #optimizer=tf.train.ProximalAdagradOptimizer(
  #    learning_rate=0.1,
  #    l1_regularization_strength=0.001,
  #    l2_regularization_strength=0.001)
  if model_type == "wide":
    m = tf.contrib.learn.LinearRegressor(model_dir=model_dir,
                                          feature_columns=wide_columns)
  
  elif model_type == "deep":
    m = tf.contrib.learn.DNNRegressor(model_dir=model_dir,
                                       feature_columns=deep_columns,
                                       hidden_units=[50,50,50,50])
  else:
    m = tf.contrib.learn.DNNLinearCombinedRegressor(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[50,50,50,50,50,50],
        #dnn_optimizer=optimizer
        )
  
  return m


def input_fn(df, LABEL_COLUMN):
  """Input builder function."""
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols)

  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values,dtype="float64")
  # Returns the feature columns and the label.
  return feature_cols, label


def train_and_eval(df_train,df_test,model_type, LABEL_COLUMN):
  """Train and evaluate the model."""

  model_dir = tempfile.mkdtemp()
  print("model directory = %s" % model_dir)
  train_steps = 2500
  
  m = build_estimator(model_dir,model_type)
  m.fit(input_fn=lambda: input_fn(df_train,LABEL_COLUMN), steps=train_steps)

  results=m.predict(input_fn=lambda: input_fn(df_test,LABEL_COLUMN))
  
  results=list(itertools.islice(results,len(df_test)))
  
  
  return results

def main(_):
  train_and_eval()


if __name__ == "__main__":
  tf.app.run()
