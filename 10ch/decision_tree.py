from housing_data import HousingData
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor

def line_regplot(X, y, model):
  plt.scatter(X, y, c='blue')
  plt.plot(X, model.predict(X), color="red")
  return None

data = HousingData()

df = data.df
X = df[['LSTAT']].values
y = df['MEDV'].values

tree = DecisionTreeRegressor(max_depth = 3)
tree.fit(X, y)
sort_idx = X.flatten().argsort()
line_regplot(X[sort_idx], y[sort_idx], tree)
plt.xlabel('% lower status of the population [LSTAT]')
plt.ylabel('Price in $1000\'s [MEDV]')
plt.show()

y_tree_pred = tree.predict(X)

from sklearn.metrics import r2_score
print('R^2 liner: %.3f' % (r2_score(y, y_tree_pred)))