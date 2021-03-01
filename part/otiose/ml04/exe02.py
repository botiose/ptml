from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

dataset = load_iris()
data = pd.DataFrame(data=np.c_[dataset.data, dataset.target],
                     columns=dataset.feature_names + ["target"])

train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

fig, axes = plt.subplots(nrows=2, ncols=2)

train_set.plot(kind="scatter", x="sepal length (cm)", y="sepal width (cm)",
               c="target", cmap="viridis", ax=axes[0][0], colorbar=False)
test_set.plot(kind="scatter", x="sepal length (cm)", y="sepal width (cm)",
               c="target", cmap="viridis", ax=axes[0][0], colorbar=False,
              alpha=0.4)
axes[0][0].set_xlabel("Train Set")
axes[0][0].set_ylabel("")

for clusterCount in range(2, 5):
    model = KMeans(n_clusters=clusterCount)
    PredCluster02 = model.fit_predict(train_set[["sepal length (cm)",
                                                 "sepal width (cm)",
                                                 "petal length (cm)",
                                                 "petal width (cm)"]])
    train_set.target = PredCluster02
    xSubplot, ySubplot = (clusterCount-1) // 2, (clusterCount-1) % 2
    train_set.plot(kind="scatter", x="sepal length (cm)", y="sepal width (cm)",
              c="target", cmap="viridis", ax=axes[xSubplot][ySubplot],
              colorbar=False)
    TestPredCluster02 = model.fit_predict(test_set[["sepal length (cm)",
                                                    "sepal width (cm)",
                                                    "petal length (cm)",
                                                    "petal width (cm)"]])
    test_set.target = TestPredCluster02
    test_set.plot(kind="scatter", x="sepal length (cm)", y="sepal width (cm)",
                  c="target", cmap="viridis", ax=axes[xSubplot][ySubplot],
                  colorbar=False, alpha=0.4)
    
    centers = model.cluster_centers_
    axes[xSubplot][ySubplot].scatter(centers[:, 0], centers[:, 1], c="red")
    axes[xSubplot][ySubplot].set_xlabel("Cluster Count: " + str(clusterCount))
    axes[xSubplot][ySubplot].set_ylabel("")

plt.show()
