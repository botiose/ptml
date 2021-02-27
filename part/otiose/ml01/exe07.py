import pandas as pd

data2 = pd.read_csv("data2.csv", sep=";")

print(data2.where(((data2.country == "Muscovy") |
            (data2.country == "Ryazan") |
            (data2.country == "Novgorod")) &
           (data2.goods != "Grain")).dev.sum())
