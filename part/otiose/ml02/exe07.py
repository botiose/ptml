from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

data1 = pd.read_csv("data1.csv")[["age", "fare", "survived", "sex"]].dropna()

d = {"male": 1, 'female': 0}
data1.sex = data1.sex.map(d)

colors = ["red" if i == 1 else "blue" for i in data1.survived]

pd.plotting.scatter_matrix(data1[["age", "fare", "sex"]], diagonal="kde",
                           color=colors);

plt.show()
