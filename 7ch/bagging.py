from winedata import WineData
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='entropy', max_depth=None, random_state=1)
bag = BaggingClassifier(base_estimator=tree, n_estimators=500, 
                        max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False,
                        n_jobs=1, random_state=1)

from sklearn.metrics import accuracy_score
data = WineData()
tree = tree.fit(data.X_train, data.y_train)
y_train_pred = tree.predict(data.X_train)
y_test_pred = tree.predict(data.X_test)
tree_train = accuracy_score(data.y_train, y_train_pred)
tree_test = accuracy_score(data.y_test, y_test_pred)
print("Decision tree train/test accuracies %.3f / %.3f" % (tree_train, tree_test))

bag = bag.fit(data.X_train, data.y_train)
y_train_pred = bag.predict(data.X_train)
y_test_pred = bag.predict(data.X_test)
bag_train = accuracy_score(data.y_train, y_train_pred)
bag_test = accuracy_score(data.y_test, y_test_pred)
print("Bagging train/test accuracies %.3f / %.3f" % (bag_train, bag_test))

import numpy as np
import matplotlib.pyplot as plt

x_min = data.X_train[:, 0].min() - 1
x_max = data.X_train[:, 0].max() + 1
y_min = data.X_train[:, 1].max() - 1
y_max = data.X_train[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

f, axarr = plt.subplots(nrows=1, ncols=2, sharex='col', sharey='row', figsize=(8, 3))

for idx, clf, tt in zip([0, 1], [tree, bag], ['Decision Tree', 'Bagging']):
  clf.fit(data.X_train, data.y_train)
  z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
  z = z.reshape(xx.shape)
  axarr[idx].contourf(xx, yy, z, alpha=0.3)
  axarr[idx].scatter(data.X_train[data.y_train==0, 0], data.X_train[data.y_train==0, 1], c='blue', marker='^')
  axarr[idx].scatter(data.X_train[data.y_train==1, 0], data.X_train[data.y_train==1, 1], c='red', marker='o')
  axarr[idx].set_title(tt)

axarr[0].set_ylabel('Alchol', fontsize=12)
plt.text(10.2, -1.2, s='Hue', ha='center', va='center', fontsize=12)
plt.show()