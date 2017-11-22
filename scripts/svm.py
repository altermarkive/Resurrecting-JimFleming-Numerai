import pandas as pd
from sklearn import model_selection
from sklearn.svm import SVC as svc
from sklearn.metrics import accuracy_score

training_data = pd.read_csv('../datasets/numerai_training_data.csv', header=0)
tournament_data = pd.read_csv('../datasets/numerai_tournament_data.csv', header=0)
features = [f for f in list(training_data) if 'feature' in f]

#this returns four arrays which is in the order of features_train, features_test, labels_train, labels_test
features_train, features_test, labels_train, labels_test = model_selection.train_test_split(training_data[features], training_data['target'], test_size=0.3, random_state=0)

clf = svc(C=1.0, probability=True).fit(features_train, labels_train)
# Alternative: calibration.CalibratedClassifierCV(svm.LinearSVC(C=1.0, verbose=True))

#predicting our target value with the 30% remnant of the training_data
predictions = clf.predict(features_test)
print(predictions)

accuracy = accuracy_score(labels_test,predictions, normalize=True, sample_weight=None)
print(accuracy)
#c = 1.0 -> 0.514361849391
#c = 100.0 -> 0.518133997785

prob_predictions_tourney = clf.predict_proba(tournament_data[features])

t_id = tournament_data['id']

results = prob_predictions_tourney[:, 1]
results_df = pd.DataFrame(data={'probability':results})
joined = pd.DataFrame(t_id).join(results_df)
joined.to_csv('predictions.ay_rfc.csv', index=False)
