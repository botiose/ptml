import numpy as np
import pandas as pd

data1 = pd.read_csv("data1.csv")

print(data1.isna().sum().sort_values(ascending=False))
