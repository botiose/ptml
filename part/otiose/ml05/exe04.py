from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from warnings import filterwarnings

filterwarnings('ignore')

fig, axes = plt.subplots(nrows=1, ncols=2)

X_iris, y_iris = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris)

axes[0].scatter(X_train[:,0], X_train[:,1], c=y_train)
axes[0].scatter(X_test[:,0], X_test[:,1], c=y_test, alpha=0.4)

model = LinearSVC()
model.fit(X_train, y_train)

axes[1].scatter(X_iris[:,0], X_iris[:,1], c=model.predict(X_iris))

plt.show()
