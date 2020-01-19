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

X_log = np.log(X)
y_sqrt = np.sqrt(y)
X_fit = np.arange(X_log.min()-1, X_log.max()+1, 1)[:, np.newaxis]
regr = regr.fit(X_log, y_sqrt)
y_line_fit = regr.predict(X_fit)
liner_r2 = r2_score(y_sqrt, regr.predict(X_log))

plt.scatter(X_log, y_sqrt, label="training points", color="lightgray")
plt.plot(X_fit, y_line_fit, label='liner (d = 1), $R^2=%.2f$' % liner_r2, color='blue', lw=2)
plt.xlabel('log(% lower status of the population [LSTAT])')
plt.ylabel('$\sqrt{ Price [MEDV]})')
plt.legend(loc="lower left")
plt.show()