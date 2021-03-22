import pandas as pd
from scipy import stats 

data1 = pd.read_csv("data1.csv")[["sex", "pclass"]].dropna()

print(stats.chi2_contingency(pd.crosstab(data1.pclass, data1.sex)))
