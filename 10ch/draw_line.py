from housing_data import HousingData
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from linear_regression import GD

data = HousingData()
df = data.df
X = df[['RM']].values
y = df[['MEDV']].values

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()

# x_std: [[a], [b], [c], [f]]
X_std = sc_x.fit_transform(X)

# y_std: [a, b, c, f...]
y_std = sc_y.fit_transform(y).flatten()

def line_regplot(X, y, model):
  plt.scatter(X, y, c='blue')
  plt.plot(X, model.predict(X), color="red")
  return None


lr = LinearRegressionGD()
lr.fit(X_std, y_std)

line_regplot(X_std, y_std, lr)

plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000\'s[MEDV] (standardized)')
#plt.show()

num_rooms_std = sc_x.transform([[5.0]])
price_std = lr.predict(num_rooms_std)
print("Price in $1000's: %.3f" % sc_y.inverse_transform(price_std))