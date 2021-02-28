from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

data1 = pd.read_csv("data1.csv")

data1.plot(kind="scatter", x="age", y="fare", c="survived", cmap="bwr",
           colorbar=False, alpha=0.4)

plt.legend(handles=[mpatches.Patch(color='red', label="Survived"),
                    mpatches.Patch(color='blue', label="Died")])

plt.show()
