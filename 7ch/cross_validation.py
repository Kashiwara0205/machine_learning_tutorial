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

clf1 = LogisticRegression(penalty='l2', C = 0.001, random_state=0)
clf2 = DecisionTreeClassifier(max_depth=1, criterion='entropy', random_state=0)
clf3 = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')
pipe1 = Pipeline([['sc', StandardScaler()], ['clf', clf1]])
pipe3 = Pipeline([['sc', StandardScaler()], ['clf', clf3]])

clf_labels= ['Logistic Regression', 'Decision Tree', 'KNN']
print('10-fold cross validation:\n')

data = IrisData()

mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])
clf_labels += ['Majority Voting']
all_clf = [pipe1, clf2, pipe3, mv_clf]

mv_clf.fit(X = data.X_train, y = data.y_train)
mv_clf.predict(X = data.X_train)

for clf, label in zip(all_clf, clf_labels):
  scores = cross_val_score(estimator=clf, X = data.X_train, y = data.y_train, cv= 10, scoring='roc_auc')
  print("ROC AUC: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))


from sklearn.metrics import roc_curve
from sklearn.metrics import auc
colors = ['black', 'orange', 'blue', 'green']
linerstyle = [':', '--', '-.', '-']
for clf, label, clr, ls in zip(all_clf, clf_labels, colors, linerstyle):
    y_pred = clf.fit(data.X_train, data.y_train).predict_proba(data.X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_true=data.y_test, y_score=y_pred)
    print("fpr", fpr.shape[0])
    print("tpr", tpr.shape[0])
    roc_auc = auc(x=fpr, y=tpr)
    plt.plot(fpr, tpr, color=clr, linestyle=ls, label='%s (auc = %0.2f)' % (label, roc_auc))

plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.grid()
plt.show()