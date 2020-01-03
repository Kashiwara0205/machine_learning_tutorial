from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from cancer_data import CancerData

pipe_svc =  Pipeline([('scl', StandardScaler()),
                      ('clf', SVC(random_state=1))])
    
param_range = [0.0001, 0.001, 0.01, 0.1, 1, 10.0, 100.0, 1000.0]
param_grid = [{'clf__C': param_range, 'clf__kernel': ['linear']},
              {'clf__C': param_range, 'clf__gamma': param_range, 'clf__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=10,
                  n_jobs=-1)

data = CancerData()
gs = gs.fit(data.X_train, data.y_train)
print(gs.best_score_)
print(gs.best_params_)
clf = gs.best_estimator_

clf.fit(data.X_train, data.y_train)
print('Test accracy: %.3f' % clf.score(data.X_test, data.y_test))