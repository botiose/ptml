import numpy as np
import pandas as pd
from scipy import stats 

data1 = pd.DataFrame(np.random.randint(0, 2, size=(100, 2)),
                     columns=['attr01', 'attr02'])

print(stats.chi2_contingency(pd.crosstab(data1.attr01, data1.attr02)))
