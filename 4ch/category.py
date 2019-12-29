from sklearn.preprocessing import LabelEncoder

import pandas as pd
df = pd.DataFrame([
 ['green', 'M', 10.1, 'class1'],
 ['red', 'L', 13.5, 'class2'],
 ['blue', 'XL', 15.3, 'class1']
])


df.columns = ['color', 'size', 'price', 'classlabel']
print(df)

# convert size to values
size_mapping = {'XL': 3, 'L': 2, 'M': 1}
df['size'] = df['size'].map(size_mapping)
print("converted size to values")
print("---------------------------------")
print(df)

import numpy as np

# convert class label to values
class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}
df['classlabel'] = df['classlabel'].map(class_mapping)
print("converted class label to values")
print("---------------------------------")
print(df)

X = df[['color', 'size', 'price']].values
print("------------------------------")
print(X)
print("------------------------------")
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
print(X)
print("------------------------------")

print("convert color to one hot")
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[0])
print(ohe.fit_transform(X).toarray())
print("------------------------------")
print(pd.get_dummies(df[['price', 'color', 'size']]))