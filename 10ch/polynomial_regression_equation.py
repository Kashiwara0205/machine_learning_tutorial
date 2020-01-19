from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

X = np.array([258.0, 270.0, 294.0, 320.0, 342.0, 368.0, 396.0, 446.0, 480.0, 586.0])[:, np.newaxis]
y = np.array([236.4, 234.4, 252.8, 298.6, 314.2, 342.2, 360.8, 368.0, 391.2, 390.8])

lr = LinearRegression()
pr = LinearRegression()

quadatic = PolynomialFeatures(degree=2)
X_quad = quadatic.fit_transform(X)

lr.fit(X, y)
X_fit = np.arange(250, 600, 10)[:, np.newaxis]
y_lin_fit = lr.predict(X_fit)

pr.fit(X_quad, y)
y_quad_fit = pr.predict(quadatic.fit_transform(X_fit))
plt.scatter(X, y, label='training points')
plt.plot(X_fit, y_lin_fit, label='linear fit', linestyle="--")
plt.plot(X_fit, y_quad_fit, label="quadratic fit")
plt.legend(loc="upper left")
plt.show()

y_lin_pred = lr.predict(X)
y_quad_pred = pr.predict(X_quad)
from sklearn.metrics import mean_squared_error
print('MSE liner: %.3f, quadratic: %.3f' % (mean_squared_error(y, y_lin_pred),
                                       mean_squared_error(y, y_quad_pred)))
                                    
from sklearn.metrics import r2_score
print('R^2 liner: %.3f, quadratic: %.3f' % (r2_score(y, y_lin_pred),
                                       r2_score(y, y_quad_pred)))
