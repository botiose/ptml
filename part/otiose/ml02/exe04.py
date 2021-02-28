from matplotlib import pyplot as plt
import pandas as pd

data1 = pd.read_csv("data1.csv")

data1.plot(kind="scatter", x="age", y="fare", alpha=0.4)

plt.show()

