from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

data1 = pd.read_csv("data1.csv")[["age", "fare", "survived"]].dropna()

colors = ["red" if i == 1 else "blue" for i in data1.survived]

pd.plotting.scatter_matrix(data1[["age", "fare"]], diagonal="kde",
                           color=colors);

plt.show()
