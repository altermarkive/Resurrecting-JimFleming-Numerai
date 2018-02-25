#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 13:40:47 2017

@author: jeremy
"""
import numpy as np
import pandas as pd
import datetime as dt
import wide_and_deep_model
import numpy as np

import os
import sys
    
    
eps = sys.float_info.epsilon

if __name__ == "__main__":
    start_time = dt.datetime.now()
    print("Start time: ",start_time)
  

    train = pd.read_csv(os.getenv('TRAINING'), header=0)
    test = pd.read_csv(os.getenv('TESTING'), header=0)

    features = [f for f in list(train) if 'feature' in f]
    test['target']=np.nan
    

    
    print(features)

    print("Building model.. ",dt.datetime.now()-start_time)
    preds = wide_and_deep_model.train_and_eval(train, test, 'wide_n_deep', features, 'target')

    
    
    print("Creating submission file...")
    out_df = test.id.copy()
    out_df=out_df.to_frame()
    out_df['probability'] = np.clip(preds, 0.0 + eps, 1.0 - eps)
    out_df.to_csv(os.getenv('PREDICTING'), index=False, float_format='%.16f')
    print("Submission file created: ") 
    
    print(dt.datetime.now()-start_time)
    
    
    
    
