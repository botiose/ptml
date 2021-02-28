from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

data1 = pd.read_csv("data1.csv")

X = data1.PetalWidth.values.reshape(-1, 1)
y = data1.PetalLength

model = LinearRegression()
model.fit(X, y)

model_x = np.linspace(min(X), max(X), 2)
model_y = model.coef_[0]*model_x + model.intercept_

data1.plot(kind="scatter", x="PetalWidth", y="PetalLength", colorbar=False,
           alpha=0.4)
plt.plot(model_x, model_y, "r-", label="Score: " +
         str(round(model.score(X, y), 2)))

plt.legend()
plt.show()
