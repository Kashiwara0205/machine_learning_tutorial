from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from cancer_data import CancerData
from sklearn.svm import SVC
import matplotlib.pyplot as plt

data = CancerData()

pipe_svc =  Pipeline([('scl', StandardScaler()),
                      ('clf', SVC(random_state=1))])



pipe_svc.fit(data.X_train, data.y_train)
y_pred = pipe_svc.predict(data.X_test)
confmat = confusion_matrix(y_true = data.y_test, y_pred = y_pred)
print(confmat)

fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
  for j in range(confmat.shape[1]):
    ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicetd label')
plt.ylabel('true label')
#plt.show()

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
print('Precision: %.3f ' % precision_score(y_true=data.y_test, y_pred=y_pred))
print('Recall: %.3f' % recall_score(y_true=data.y_test, y_pred=y_pred))

print('F1: %.3f' % f1_score(y_true=data.y_test, y_pred=y_pred))

