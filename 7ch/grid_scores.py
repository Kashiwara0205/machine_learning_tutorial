from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from majority_vote_classifier import MajorityVoteClassifier
import matplotlib.pyplot as plt
from iris_data import IrisData
from sklearn.model_selection import GridSearchCV
data = IrisData()
clf1 = LogisticRegression(penalty='l2', C = 0.001, random_state=0)
clf2 = DecisionTreeClassifier(max_depth=1, criterion='entropy', random_state=0)
clf3 = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')
pipe1 = Pipeline([['sc', StandardScaler()], ['clf', clf1]])
pipe3 = Pipeline([['sc', StandardScaler()], ['clf', clf3]])

mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])

params = { 'decisiontreeclassifier__max_depth': [1, 2],
            'pipeline-1__clf__C': [0.001, 0.1, 100.0] }

grid = GridSearchCV(estimator=mv_clf, param_grid=params, cv=10, scoring='roc_auc')
grid.fit(data.X_train, data.y_train)

means = grid.cv_results_['mean_test_score']
params = grid.cv_results_['params']

for mean, param in zip(means,params):
  print("%f  with:   %r" % (mean,param))

mv_clf.get_params(deep=False)