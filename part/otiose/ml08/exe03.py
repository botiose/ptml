import numpy as np
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt

X_iris, _ = load_iris(return_X_y=True)
petalLength, petalWidth = X_iris[:,2], X_iris[:,3]
plt.scatter(petalLength, petalWidth)

bias = np.ones((len(petalLength), 1))
X = np.c_[bias, petalLength]
Y = petalWidth
# Normal equation: closed form solution for MSE
paramsMSE = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(Y))
predictMSE = paramsMSE[1]*X_iris+paramsMSE[0]
plt.plot(X_iris, predictMSE, "r-")

bs = np.linspace(-1, 1, 100)
ms = np.linspace(-1, 1, 100)

min_score = float("inf")
min_b, min_m = 0, 0

for b in bs:
    for m in ms:
        score = sum(abs(m*petalLength+b - Y))/len(X_iris)
        if min_score > score:
            min_b, min_m = b, m
            min_score = score

predictMAD = min_m*X_iris+min_b
plt.plot(X_iris, predictMAD, "b-")
plt.show()
