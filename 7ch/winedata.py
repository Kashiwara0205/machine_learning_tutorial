import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class WineData():
  def __init__(self):
    df_wine = pd.read_csv(
      'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
      header=None)
    df_wine.columns = ['Class label', 'Alchol', 'Malic acid', 'Ash', 'Alcalinity of ash',
                       'Magnesium', 'Total phenols', 'Flavanoids', 'Nounflavanoid phenols', 
                       'Proanthocyanins', 'Color intensity','Hue', 'OD280/OD315 of dilluted wines', 'Proline']
    df_wine = df_wine[df_wine['Class label'] != 1]
    y = df_wine['Class label'].values
    X = df_wine[['Alchol', 'Hue']].values
    le = LabelEncoder()
    y = le.fit_transform(y)
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.40, random_state=1)