from housing_data import HousingData
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
import numpy as np

data = HousingData()

ransac = RANSACRegressor(LinearRegression(), 
                         max_trials=100, 
                         min_samples=50, 
                         residual_threshold=5.0, 
                         random_state=0)

ransac.fit(data.X, data.y)

inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
line_x = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_x[:, np.newaxis])

plt.scatter(data.X[inlier_mask], data.y[inlier_mask], c = 'blue', marker='o', label='Inliers')
plt.scatter(data.X[outlier_mask], data.y[outlier_mask], c = 'lightgreen', marker='s', label='Outliers')

plt.plot(line_x, line_y_ransac, color='red')
plt.xlabel('Average number of romms[RM]')
plt.ylabel('Price in $1000\'s [MEDV]')
plt.legend(loc='upper left')
plt.show()