import numpy as np
from matplotlib import pyplot as plt

xData=np.linspace(-1, 1, 10)
yData=np.linspace(-10, 10, 10)
plt.scatter(xData, yData)

bias = np.ones((len(xData), 1))
X = np.c_[bias, xData]
Y = yData
# Normal equation: closed form solution for MSE
paramsMSE = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(Y))
predictMSE = paramsMSE[1]*xData+paramsMSE[0]
plt.plot(xData, predictMSE, "r-")

bs = np.linspace(-1, 1, 100)
ms = np.linspace(-1, 1, 100)

min_score = float("inf")
min_b, min_m = 0, 0

for b in bs:
    for m in ms:
        score = sum(abs(m*xData+b - Y))/len(xData)
        if min_score > score:
            min_b, min_m = b, m
            min_score = score

predictMAD = min_m*xData+min_b
plt.plot(xData, predictMAD, "b-")
plt.show()
