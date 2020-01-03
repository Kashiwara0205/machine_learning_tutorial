from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from cancer_data import CancerData
from sklearn.model_selection import cross_val_score
from cancer_data import CancerData
import numpy as np

data = CancerData()

pipe_svc =  Pipeline([('scl', StandardScaler()),
                      ('clf', SVC(random_state=1))])
    
param_range = [0.0001, 0.001, 0.01, 0.1, 1, 10.0, 100.0, 1000.0]
param_grid = [{'clf__C': param_range, 'clf__kernel': ['linear']},
              {'clf__C': param_range, 'clf__gamma': param_range, 'clf__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=2,
                  n_jobs=-1)

scores = cross_val_score(gs, data.X_train, data.y_train, scoring='accuracy', cv=5)
print('SVC -> CV accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

from sklearn.tree import DecisionTreeClassifier
gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),
                  param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}],
                  cv=2)

scores = cross_val_score(gs, 
                         data.X_train,
                         data.y_train,
                         scoring='accuracy',
                         cv=5)

print('DecisionTree -> CV accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))
