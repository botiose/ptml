import pandas as pd

data1 = pd.read_csv("data1.csv")

print(data1.groupby("pclass").agg({"pclass" : "count",
                                   "age" : "mean",
                                   "survived" : "mean"}))
