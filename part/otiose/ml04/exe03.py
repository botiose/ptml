from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd

dataset = load_iris()
data = pd.DataFrame(data=np.c_[dataset.data, dataset.target],
                     columns=dataset.feature_names + ["target"])

fig, axes = plt.subplots(nrows=2, ncols=2)

data.plot(kind="scatter", x="sepal length (cm)", y="sepal width (cm)",
          c="target", cmap="viridis", ax=axes[0][0], colorbar=False)
axes[0][0].set_xlabel("Data")
axes[0][0].set_ylabel("")

for clusterCount in range(2, 5):
    model = AgglomerativeClustering(n_clusters=clusterCount)
    PredCluster02 = model.fit_predict(data[["sepal length (cm)",
                                            "sepal width (cm)",
                                            "petal length (cm)",
                                            "petal width (cm)"]])
    data.target = PredCluster02
    xSubplot, ySubplot = (clusterCount-1) // 2, (clusterCount-1) % 2
    data.plot(kind="scatter", x="sepal length (cm)", y="sepal width (cm)",
              c="target", cmap="viridis", ax=axes[xSubplot][ySubplot],
              colorbar=False)
    axes[xSubplot][ySubplot].set_xlabel("Cluster Count: " + str(clusterCount))
    axes[xSubplot][ySubplot].set_ylabel("")

plt.show()
