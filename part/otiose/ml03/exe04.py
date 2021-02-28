from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd

data2 = pd.read_csv("data2.csv", sep=";")[["adm", "dip", "mil"]].dropna()

model = PCA()
model.fit_transform(data2)

# Show the preserved variance of each principal component as a result
print(model.explained_variance_ratio_)
