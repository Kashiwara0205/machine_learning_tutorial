from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

class IrisData():
  def __init__(self):
    iris = datasets.load_iris()
    X, y = iris.data[50:, [1, 2]], iris.target[50:]
    le = LabelEncoder()
    y = le.fit_transform(y)

    self.X_train, self.X_test, self.y_train, self.y_test = \
      train_test_split(X, y, test_size=0.5, random_state=1)