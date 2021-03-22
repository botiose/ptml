import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from scipy import stats

dataset = load_iris()
data = pd.DataFrame(data=np.c_[dataset.data], columns=dataset.feature_names)
petalLength = data["petal length (cm)"] 

subSampleMean = petalLength.sample(30).mean()

print(stats.ttest_1samp(petalLength.to_numpy(), subSampleMean))
