from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

def predict(x_label, y_label, subPlotIndex):
    X = data1[x_label].values.reshape(-1, 1)
    y = data1[y_label]

    model = LinearRegression()
    model.fit(X, y)

    model_x = np.linspace(min(X), max(X), 2)
    model_y = model.coef_[0]*model_x + model.intercept_

    axes[subPlotIndex].plot(model_x, model_y, "r-", label="Score: " +
                            str(round(model.score(X, y), 2)))
    data1.plot(kind="scatter", x=x_label, y=y_label, colorbar=False,
               alpha=0.4, ax=axes[subPlotIndex])
    axes[subPlotIndex].legend()

    return subPlotIndex + 1

data1 = pd.read_csv("data1.csv")
fig, axes = plt.subplots(nrows=1, ncols=2)
subPlotIndex = 0

subPlotIndex = predict("PetalWidth", "PetalLength", subPlotIndex)
predict("PetalWidth", "SepalLength", subPlotIndex)

plt.show()
