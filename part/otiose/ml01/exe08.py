import pandas as pd

data2 = pd.read_csv("data2.csv", sep=";")

exe07 = data2.where(((data2.country == "Muscovy") |
                     (data2.country == "Ryazan") |
                     (data2.country == "Novgorod")) &
                    (data2.goods != "Grain"))

print(exe07.where(exe07.goods == "Fur").dropna(how="all").dev
      .apply(lambda x: max(3,x-5)).sum())
