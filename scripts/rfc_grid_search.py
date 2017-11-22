# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.grid_search import GridSearchCV as GS
from sklearn.metrics import accuracy_score, log_loss

training_data = pd.read_csv('../datasets/numerai_training_data.csv', header=0)
tournament_data = pd.read_csv('../datasets/numerai_tournament_data.csv', header=0)
features = [f for f in list(training_data) if 'feature' in f]

# splitting my arrays in ratio of 30:70 percent
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(training_data[features], training_data['target'], test_size=0.3, random_state=0)

# parameters
parameters = {
        'n_estimators': [ 20,25 ],
        'random_state': [ 0 ],
        'max_features': [ 2 ],
        'min_samples_leaf': [150,200,250]
}

# implementing my classifier
model = RFC()
grid = GS(estimator=model, param_grid=parameters)
grid.fit(features_train, labels_train)

# Calculate the logloss of the model
prob_predictions_class_test = grid.predict(features_test)
prob_predictions_test = grid.predict_proba(features_test)

logloss = log_loss(labels_test,prob_predictions_test)

accuracy = accuracy_score(labels_test, prob_predictions_class_test, normalize=True,sample_weight=None)

# predict class probabilities for the tourney set
prob_predictions_tourney = grid.predict_proba(tournament_data[features])

t_id = tournament_data['id']

results = prob_predictions_tourney[:, 1]
results_df = pd.DataFrame(data={'probability':results})
joined = pd.DataFrame(t_id).join(results_df)
joined.to_csv('predictions.ay_rfc_grid.csv', index=False)
