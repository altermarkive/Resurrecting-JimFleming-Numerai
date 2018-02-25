import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import accuracy_score, log_loss

import os
import sys

eps = sys.float_info.epsilon

training_data = pd.read_csv(os.getenv('TRAINING'), header=0)
tournament_data = pd.read_csv(os.getenv('TESTING'), header=0)
features = [f for f in list(training_data) if 'feature' in f]

# splitting my arrays in ratio of 30:70 percent
features_train, features_test, labels_train, labels_test = model_selection.train_test_split(training_data[features], training_data['target'], test_size=0.3, random_state=0)

# implementing my classifier
clf = RFC(n_estimators=25, random_state=0, n_jobs=-1).fit(features_train, labels_train)

# Calculate the logloss of the model
prob_predictions_class_test = clf.predict(features_test)
prob_predictions_test = clf.predict_proba(features_test)

logloss = log_loss(labels_test,prob_predictions_test)
accuracy = accuracy_score(labels_test, prob_predictions_class_test, normalize=True,sample_weight=None)
print('accuracy', accuracy)
print('logloss', logloss)

# predict class probabilities for the tourney set
prob_predictions_tourney = clf.predict_proba(tournament_data[features])

t_id = tournament_data['id']

results = prob_predictions_tourney[:, 1]
results_df = pd.DataFrame(data={'probability':results})
joined = pd.DataFrame(t_id).join(np.clip(results_df, 0.0 + eps, 1.0 - eps))
joined.to_csv(os.getenv('PREDICTING'), index=False, float_format='%.16f')
