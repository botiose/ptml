import pandas as pd

data1 = pd.read_csv("data1.csv")

print(data1.where(data1.pclass == 1).dropna(how="all"))
