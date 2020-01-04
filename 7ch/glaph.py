from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from iris_data import IrisData
from majority_vote_classifier import MajorityVoteClassifier
import matplotlib.pyplot as plt
from iris_data import IrisData

clf1 = LogisticRegression(penalty='l2', C = 0.001, random_state=0)
clf2 = DecisionTreeClassifier(max_depth=5, criterion='entropy', random_state=0)
clf3 = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')
pipe1 = Pipeline([['sc', StandardScaler()], ['clf', clf1]])
pipe3 = Pipeline([['sc', StandardScaler()], ['clf', clf3]])

clf_labels= ['Logistic Regression', 'Decision Tree', 'KNN']
data = IrisData()

mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])
clf_labels += ['Majority Voting']
all_clf = [pipe1, clf2, pipe3, mv_clf]

sc = StandardScaler()
X_train_std = sc.fit_transform(data.X_train)

from itertools import product
x_min = X_train_std[:, 0].min()
x_max = X_train_std[:, 0].max()
y_min = X_train_std[:, 1].min()
y_max = X_train_std[:, 1].max()

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

f, axarr = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row', figsize=(7, 5))

for idx, clf, tt in zip(product([0, 1], [0, 1]), all_clf, clf_labels):
  clf.fit(X_train_std, data.y_train)
  z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
  z = z.reshape(xx.shape)
  
  axarr[idx[0], idx[1]].contourf(xx, yy, z, alpha=0.3)
  axarr[idx[0], idx[1]].scatter(X_train_std[data.y_train==0, 0], X_train_std[data.y_train==0, 1], c='blue', marker='^', s=50)
  axarr[idx[0], idx[1]].scatter(X_train_std[data.y_train==1, 0], X_train_std[data.y_train==1, 1], c='red', marker='o', s=50)
  axarr[idx[0], idx[1]].set_title(tt)

plt.text(-3.5, -4.5, s='Speal width [standardized]', ha='center', va='center', fontsize=12)
plt.text(-10.5, 4.5, s='Petal length [standardized]', ha='center', va='center', fontsize=12, rotation=90)
plt.show()