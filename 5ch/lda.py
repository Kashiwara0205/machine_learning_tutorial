# -*- coding: utf-8 -*-

import pandas as pd

df_wine = pd.read_csv(
'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
header=None)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

import numpy as np

np.set_printoptions(precision=4)
mean_vecs = []

for label in range(1, 4):
  mean_vecs.append(np.mean(X_train_std[y_train==label], axis=0))
  print("MV %s: %s\n" %(label, mean_vecs[label -1 ]))

d = 13
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
  class_scatter = np.zeros((d, d))
  for row in X_train_std[y_train == label]:
    row, mv = row.reshape(d, 1), mv.reshape(d, 1)
    class_scatter += (row-mv).dot((row-mv).T)
  S_W += class_scatter

print("Within-class scatter matrix: %sx%s" % (S_W.shape[0], S_W.shape[1]))

# クラス内変動行列(共分散行列)を生成
d = 13
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
  class_scatter = np.cov(X_train_std[y_train==label].T)
  S_W += class_scatter

# クラス間変動行列(共分散行列)を生成
mean_overall = np.mean(X_train_std, axis=0)
d = 13
S_B = np.zeros((d, d))
for i, mean_vec in enumerate(mean_vecs):
  n = X_train[y_train==i+1, :].shape[0]
  mean_vec = mean_vec.reshape(d, 1)
  S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)


# 固有値の計算
eign_vals, eign_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

eign_pairs = [(np.abs(eign_vals[i]), eign_vecs[:, i])
for i in range(len(eign_vals))]

eign_pairs = sorted(eign_pairs, key=lambda k: k[0], reverse=True)
for eign_val in eign_pairs:
  print(eign_val[0])

tot = sum(eign_vals.real)

discr = [(i/tot) for i in sorted(eign_vals.real, reverse=True)]
cum_discr = np.cumsum(discr)

import matplotlib.pyplot as plt
# plt.bar(range(1, 14), discr, alpha=0.5, align='center', label='individual "discrimnablility"')
# plt.step(range(1, 14), cum_discr, where="mid", label='cumulative "discriminablity"')
# plt.ylim([-0.1, 1.1])
# plt.legend(loc='best')
# plt.show()

# 変換行列を生成して射像する
w = np.hstack((eign_pairs[0][1][:, np.newaxis].real, eign_pairs[1][1][:, np.newaxis].real))
X_train_lda = X_train_std.dot(w)

colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']

for l, c, m in zip(np.unique(y_train), colors, markers):
  plt.scatter(X_train_lda[y_train==l, 0] * (-1), X_train_lda[y_train==l, 1] * (-1), c=c, label=l, marker=m)

plt.xlabel('LDA 1')
plt.ylabel('LDA 2')
plt.legend(loc='lower left')
plt.show()