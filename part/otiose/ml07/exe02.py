import numpy as np
from scipy import stats

standardNormalData = np.random.normal(size=1000)
print(stats.ttest_1samp(standardNormalData, 0))
