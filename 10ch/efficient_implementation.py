from housing_data import HousingData
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
data = HousingData()

def line_regplot(X, y, model):
  plt.scatter(X, y, c='blue')
  plt.plot(X, model.predict(X), color="red")
  return None

slr = LinearRegression()
slr.fit(data.X, data.y)
line_regplot(data.X, data.y, slr)
plt.xlabel('Average number of rooms [RM]')
plt.ylabel("Price in $1000\s [MEDV]")

plt.show()