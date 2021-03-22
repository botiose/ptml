import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.datasets import load_digits
from warnings import filterwarnings
from matplotlib import pyplot as plt
filterwarnings('ignore')

x = [i for i in range(1, 7)]
param_grid = [{'degree': x}]

model = svm.SVC(kernel="poly")
grid_search = GridSearchCV(model, param_grid, return_train_score=True)

X_digits, y_digits = load_digits(return_X_y=True)
grid_search.fit(X_digits, y_digits)

plt.plot(x, grid_search.cv_results_["mean_test_score"])
plt.xlabel("Hyperparameter: degree")
plt.ylabel("score")
plt.show()
