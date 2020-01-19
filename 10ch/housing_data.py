import pandas as pd
from sklearn.preprocessing import StandardScaler
class HousingData():
  def __init__(self):
    self.df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data", header=None, sep="\s+")
    self.df.columns = ['CRIM', 'ZM', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    df = self.df
    self.X = df[['RM']].values
    self.y = df[['MEDV']].values
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    # x_std: [[a], [b], [c], [f]]
    self.X_std = sc_x.fit_transform(self.X)
    # y_std: [a, b, c, f...]
    self.y_std = sc_y.fit_transform(self.y).flatten()