import matplotlib.pyplot as plt
from adaline_sgd import AdalineSGD
import pandas as pd
import numpy as np
import os, sys
sys.path.append(os.getcwd() + "/../util")
from decision_boudary import plot_decision_regions

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                 header=None)
                 
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values

X_std = np.copy(X)


ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada.fit(X_std, y)
plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('speal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average cost')

plt.show()