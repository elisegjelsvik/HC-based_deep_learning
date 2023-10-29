# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 16:23:34 2023

@author: Elise Lunde Gjelsvik
email: elise.lunde.gjelsvik@nmbu.no

Code used to create simulated data.
"""
from sklearn.datasets import make_blobs

X, c = make_blobs(n_samples=1500, centers=3, n_features=20, random_state=0)

y(X) = 10 * sin(pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 + 10 * X[:, 3] + 5 * X[:, 4] + noise * N(0, 1)
y(X) = (X[:, 0] ** 2 + (X[:, 1] * X[:, 2]  - 1 / (X[:, 1] * X[:, 3])) ** 2) ** 0.5 + noise * N(0, 1)
y(X) = arctan((X[:, 1] * X[:, 2] - 1 / (X[:, 1] * X[:, 3])) / X[:, 0]) + noise * N(0, 1)