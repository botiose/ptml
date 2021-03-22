from matplotlib import pyplot as plt
import numpy as np

X = np.linspace(0, 10, 10, dtype=np.int32)
y = 2*X+3

X_train, y_train = X[:8], y[:8]
