#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 13:40:47 2017

@author: jeremy
"""
import pandas as pd
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from operator import itemgetter
import xgboost as xgb
import random
import time
from sklearn.metrics import average_precision_score
from sklearn import preprocessing

from sklearn.metrics import roc_curve, auc,recall_score,precision_score

import os



    
def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()


def get_importance(gbm, features):
    create_feature_map(features)
    importance = gbm.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=itemgetter(1), reverse=True)
    return importance

def get_features(train, test):
    trainval = list(train.columns.values)
    output = trainval
    return sorted(output)


def run_single(train, test, features, target, random_state=0):
    eta = 0.1
    max_depth= 5 
    subsample = 1
    colsample_bytree = 1
    min_chil_weight=1
    start_time = time.time()

    print('XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample, colsample_bytree))
    params = {
        "objective": "reg:logistic",
        "booster" : "gbtree",
        "eval_metric": "logloss",
        "eta": eta,
        "tree_method": 'exact',
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "silent": 1,
        "min_chil_weight":min_chil_weight,
        "seed": random_state,
        #"num_class" : 22,
    }
    num_boost_round = 500
    early_stopping_rounds = 20
    test_size = 0.1

   
    
    X_train, X_valid = train_test_split(train, test_size=test_size, random_state=random_state)
    print('Length train:', len(X_train.index))
    print('Length valid:', len(X_valid.index))
    y_train = X_train[target]
    y_valid = X_valid[target]
    dtrain = xgb.DMatrix(X_train[features], y_train, missing=-99)
    dvalid = xgb.DMatrix(X_valid[features], y_valid, missing =-99)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=True)

    print("Validating...")
    check = gbm.predict(xgb.DMatrix(X_valid[features]), ntree_limit=gbm.best_iteration+1)
    
    #area under the precision-recall curve
    score = mean_squared_error(X_valid[target].values, check)
    print('mean_squared_error: {:.6f}'.format(score))

    
    check2=check.round()
    score = precision_score(X_valid[target].values, check2)
    print('precision score: {:.6f}'.format(score))

    score = recall_score(X_valid[target].values, check2)
    print('recall score: {:.6f}'.format(score))
    
    imp = get_importance(gbm, features)
    print('Importance array: ', imp)

    print("Predict test set... ")
    test_prediction = gbm.predict(xgb.DMatrix(test[features],missing = -99), ntree_limit=gbm.best_iteration+1)
    score = mean_squared_error(test[target].values, test_prediction)

    print('mean squared error test set: {:.6f}'.format(score))
    


    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
    return test_prediction, imp, gbm.best_iteration+1
    
    
    
    
    
    
    
    
    
    
    
if __name__ == "__main__":
    start_time = dt.datetime.now()
    print("Start time: ",start_time)
  

    train = pd.read_csv(os.getenv('TRAINING'), header=0)
    print(train.isnull().values.any())
    features = [f for f in list(train) if 'feature' in f]
    
    #rem_list=['feature19','feature12','feature21','feature24']
    
    #for each in rem_list:
    #    features.remove(each)
    #print(train['target'])
 
    train, test = train_test_split(train, test_size=.1, random_state=random.seed(2016))


    print(features)

    print("Building model.. ",dt.datetime.now()-start_time)
    preds, imp, num_boost_rounds = run_single(train, test, features, 'target',42)
 
    print(dt.datetime.now()-start_time)
    
    
    
    
