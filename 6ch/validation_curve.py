import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
from cancer_data import CancerData
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

data = CancerData()

pipe_lr = Pipeline([('scl', StandardScaler()), 
                    ('clf', LogisticRegression(penalty='l2', C=0.5, random_state=0))])
param_range = [0.001, 0.01, 0.1, 10.0, 100.0, 1000.0, 10000.0, 100000.0]

train_scores, test_scores = validation_curve(estimator=pipe_lr, 
                                             X= data.X_train, 
                                             y = data.y_train, 
                                             param_name = 'clf__C',
                                             param_range=param_range,
                                             cv=10)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

test_mean = np.mean(test_scores, axis=1)
test_std = np.std(train_scores, axis=1)
plt.plot(param_range, train_mean, color='blue', marker='o', markersize=5, label='training accracy')

plt.fill_between(param_range, 
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha = 0.15, color='blue')

plt.plot(param_range, test_mean, color='green', linestyle='--', 
         marker='s', markersize=5, label='validation accracy')

plt.fill_between(param_range, 
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha = 0.15, color='green')



plt.grid()
plt.xscale('log')
plt.xlabel('Parameter C')
plt.ylabel('Accracy')
plt.legend(loc='lower right')
plt.ylim([0.8, 1.0])
plt.show()