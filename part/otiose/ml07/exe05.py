import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

model = LinearRegression()

x = np.linspace(0, 100, 100)

scores = []

for s in x:
    y = 3*x+np.random.normal(0, s, 100)
    model.fit(x.reshape(-1, 1), y)
    scores.append(model.score(x.reshape(-1, 1), y))

plt.xlabel("variance")
plt.ylabel("score")
plt.plot(x, scores)
plt.show()
