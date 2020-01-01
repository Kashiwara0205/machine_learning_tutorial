import numpy as np
from sklearn.model_selection import cross_val_score
from cancer_data import CancerData
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

data = CancerData()

pipe_lr = Pipeline([('scl', StandardScaler()), 
                    ('pca', PCA(n_components= 2)),
                    ('clf', LogisticRegression(random_state=1))])

scores = cross_val_score(estimator=pipe_lr, X=data.X_train, y=data.y_train, cv=10, n_jobs=1)
print('CV accuracy scores: %s' % scores)

print('CV accracy %.3f +  / - %.3f' % (np.mean(scores), np.std(scores)))

