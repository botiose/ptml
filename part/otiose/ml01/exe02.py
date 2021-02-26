import pandas as pd

data1 = pd.read_csv("data1.csv")

print(data1.dropna(axis=1).head())
