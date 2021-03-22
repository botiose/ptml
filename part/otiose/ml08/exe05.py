from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

dataset = load_iris()
X, y = dataset.data, dataset.target
model = KMeans(n_clusters=3)

confusionMatrix = np.zeros((3,3))

for y_pred, y_label in zip(model.fit_predict(X), y):
    confusionMatrix[y_label][y_pred] += 1

ax = plt.figure().add_subplot(111)
ax.matshow(confusionMatrix, cmap=plt.cm.gray)
labels = np.ndarray.tolist(dataset.target_names)
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
plt.xlabel("Prediction")
plt.ylabel("Labels")
plt.show()
