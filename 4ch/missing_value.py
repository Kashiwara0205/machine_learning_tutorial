# -*- coding: utf-8 -*-
import pandas as pd
from io import StringIO

csv_data = '''A,B,C,D
1.0,2.0,3.0, 4.0
5.0,6.0,,8.0
10.0,11.0,12.0,
'''

csv_data = unicode(csv_data)
df = pd.read_csv(StringIO(csv_data))

print(df)
print("- - - - - -")
print(df.isnull().sum())

print("- - - - - - -")
# 行のみ
dropped_df = df.dropna()
print(dropped_df)
print("- - - - - - -")
# 列のみ
dropped_df = df.dropna(axis = 1)
print(dropped_df)
print("- - - - - - -")
# すべてがNan
dropped_df = df.dropna(how='all')
print(dropped_df)
print("- - - - - - -")
# Nanが4つ
dropped_df = df.dropna(thresh=4)
print(dropped_df)
print("- - - - - - -")
# 特定の列 CにNaNが含まれている
dropped_df = df.dropna(subset=['C'])
print(dropped_df)
print("- - - - - - -")
# 平均値保管
from sklearn.preprocessing import Imputer
imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr = imr.fit(df)
imputed_data  = imr.transform(df.values)
print(imputed_data)