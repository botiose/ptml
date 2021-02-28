from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

data1 = pd.read_csv("data1.csv")[["age", "fare", "survived", "sex"]].dropna()

d = {"male": 1, 'female': 0}
data1.sex = data1.sex.map(d)
data1.sex = data1.sex.apply(lambda x: x + np.random.uniform(0, 0.1))

colors = ["red" if i == 1 else "blue" for i in data1.survived]

pd.plotting.scatter_matrix(data1[["age", "fare", "sex"]], diagonal="kde",
                           color=colors);

plt.show()
