from cancer_data import CancerData
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

data = CancerData()

pipe_lr = Pipeline([('scl', StandardScaler()), 
                    ('pca', PCA(n_components= 2)),
                    ('clf', LogisticRegression(random_state=1))])
pipe_lr.fit(data.X_train, data.y_train)

print('Test Accuracy: %.3f' % pipe_lr.score(data.X_test, data.y_test))