import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
class CancerData():
  def __init__(self):
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', 
                      header=None)
    
    self.X =  df.loc[:, 2:].values
    self.y = df.loc[:, 1].values
    le = LabelEncoder()
    self.y = le.fit_transform(self.y)
    le.transform(['M', 'B'])
    self.X_train, self.X_test, self.y_train, self.y_test = \
      train_test_split(self.X, self.y, test_size=0.20, random_state=1)