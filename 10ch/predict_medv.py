from housing_data import HousingData
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from linear_regression import LinearRegressionGD

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
lr = LinearRegressionGD()
lr.fit(X_std, y_std)

plt.plot(range(1, lr.n_iter+1), lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')

plt.show()