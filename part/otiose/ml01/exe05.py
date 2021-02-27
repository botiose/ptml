import pandas as pd

data1 = pd.read_csv("data1.csv")

print(data1.where((30 <= data1.age) & (data1.age <= 50)).dropna(how="all"))
