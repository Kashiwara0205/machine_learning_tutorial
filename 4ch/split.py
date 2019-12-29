import pandas as pd
import numpy as np

df_wine = pd.read_csv(
'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
'Alcalinit of ash', 'Magnesium', '1', '2', '3', '4', '5', '6', '7', '8']

print('Class labels', np.unique(df_wine['Class label']))

print(df_wine.head())

from sklearn.model_selection import train_test_split
X, y = df_wine.iloc[:, 1].values, df_wine.iloc[:, 0].values

# split data to be 30 percent from whole data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print(len(X_train))
print(len(X_test))