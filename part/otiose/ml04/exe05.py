from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

data1 = pd.read_csv("data1.csv")[["age", "fare"]].dropna()

minMaxScaler = MinMaxScaler()

data1[["age", "fare"]] = minMaxScaler.fit_transform(data1[["age", "fare"]])

fig, axes = plt.subplots(nrows=2, ncols=2)

for i in range(2):
    clusterCount = 4 + i
    model = KMeans(n_clusters=clusterCount)
    PredCluster02 = model.fit_predict(data1[["age", "fare",]])
    data1["target"] = PredCluster02
    data1.plot(kind="scatter", x="age", y="fare",c="target", cmap="viridis",
              ax=axes[0][i], colorbar=False)
    centers = model.cluster_centers_
    axes[0][i].scatter(centers[:, 0], centers[:, 1], c="red")
    axes[0][i].set_xlabel("KMeans: Cluster Count: " + str(clusterCount))
    axes[0][i].set_ylabel("")

for i in range(2):
    clusterCount = 4 + i
    model = AgglomerativeClustering(n_clusters=clusterCount)
    PredCluster02 = model.fit_predict(data1[["age", "fare",]])
    data1["target"] = PredCluster02
    data1.plot(kind="scatter", x="age", y="fare",c="target", cmap="viridis",
              ax=axes[1][i], colorbar=False)
    axes[1][i].set_xlabel("HCA: Cluster Count: " + str(clusterCount))
    axes[1][i].set_ylabel("")
    
plt.show()

