from sklearn.tree import DecisionTreeClassifier
import os, sys
sys.path.append(os.getcwd() + "/../util")
from decision_boudary import plot_decision_regions
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.3, random_state=0)


from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(criterion='entropy', n_estimators=10, random_state=1, n_jobs=2)

forest.fit(X_train, y_train)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined, y_combined, classifier=forest, test_idx=range(105, 150))

plt.xlabel('petal legth[cm]')
plt.ylabel('petal width[cm]')
plt.legend(loc='upper left')

plt.xlim([0, 10])
plt.ylim([-1, 3])
plt.show()