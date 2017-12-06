#!/usr/bin/env python3
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


def build_estimator(model_dir, model_type, feature_cols):
  """Build an estimator."""

  # Wide columns and deep columns.
  wide_columns = []
  deep_columns = []
  # Continuous base columns.
  for col in feature_cols:
      feature = tf.contrib.layers.real_valued_column(col)
      wide_columns.append(feature)
      deep_columns.append(feature)

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
        fix_global_step_increment_bug=True,
        #dnn_optimizer=optimizer
        )
  
  return m


def input_fn(df, feature_cols, LABEL_COLUMN):
  """Input builder function."""
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values) for k in feature_cols}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols)

  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values,dtype="float64")
  # Returns the feature columns and the label.
  return feature_cols, label


def train_and_eval(df_train, df_test, model_type, feature_cols, LABEL_COLUMN):
  """Train and evaluate the model."""

  model_dir = tempfile.mkdtemp()
  print("model directory = %s" % model_dir)
  train_steps = 2500
  
  m = build_estimator(model_dir, model_type, feature_cols)
  m.fit(input_fn=lambda: input_fn(df_train, feature_cols, LABEL_COLUMN), steps=train_steps)

  results=m.predict(input_fn=lambda: input_fn(df_test, feature_cols, LABEL_COLUMN))
  
  results=list(itertools.islice(results,len(df_test)))
  
  
  return results

def main(_):
  train_and_eval()


if __name__ == "__main__":
  tf.app.run()
