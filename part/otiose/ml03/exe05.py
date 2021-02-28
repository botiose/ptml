from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

data2 = pd.read_csv("data2.csv", sep=";")[["adm", "dip", "mil"]].dropna()

model = PCA(n_components = 2)
X = model.fit_transform(data2)

c = np.matrix.round(model.components_, 2)

plt.xlabel("PC1: " + str(c[0][0]) + "adm + " + str(c[0][1]) + "dip + " +
           str(c[0][2]) + "mil = 0")
plt.ylabel("PC2: " + str(c[1][0]) + "adm + " + str(c[1][1]) + "dip + " +
           str(c[1][2]) + "mil = 0")
plt.scatter(*zip(*X))

plt.show()


