from housing_data import HousingData
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn.metrics import r2_score

data = HousingData()

df = data.df
X = df[['LSTAT']].values
y = df['MEDV'].values
regr = LinearRegression()

quadatic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
X_quad = quadatic.fit_transform(X)
X_cubic = cubic.fit_transform(X)

X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]
regr = regr.fit(X, y)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y, regr.predict(X))

regr = regr.fit(X_quad, y)
y_quad_fit = regr.predict(quadatic.fit_transform(X_fit))
quadatic_r2 = r2_score(y, regr.predict(X_quad))

regr = regr.fit(X_cubic, y)
y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
cubic_r2 = r2_score(y, regr.predict(X_cubic))

plt.scatter(X, y, label="training points", color="lightgray")
plt.plot(X_fit, y_lin_fit, label="liner (d = 1), $R^2$" %
         linear_r2, color="blue", lw=2, linestyle=":")

plt.plot(X_fit, y_quad_fit, label="liner (d = 2), $R^2$" %
         quadatic_r2, color="red", lw=2, linestyle="-")

plt.plot(X_fit, y_cubic_fit, label="liner (d = 3), $R^2$" %
         cubic_r2, color="green", lw=2, linestyle="--")

plt.xlabel('% lower status of population [LSTAT]')
plt.ylabel('% Price in $1000\s [MEDV]')
plt.legend(loc="upper right")
plt.show()