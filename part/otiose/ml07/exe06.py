import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.datasets import load_digits
from warnings import filterwarnings
from matplotlib import pyplot as plt
filterwarnings('ignore')

x = [ 1*10**-i for i in range(10, 0, -1)]
param_grid = [{'C': x}]

model = svm.SVC(kernel="linear")
grid_search = GridSearchCV(model, param_grid, return_train_score=True)

X_digits, y_digits = load_digits(return_X_y=True)
grid_search.fit(X_digits, y_digits)

plt.semilogx(x, grid_search.cv_results_["mean_test_score"])
plt.xlabel("Hyperparameter: C")
plt.ylabel("score")
plt.show()
