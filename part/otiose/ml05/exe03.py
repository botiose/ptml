from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from warnings import filterwarnings

filterwarnings('ignore')

X_iris, y_iris = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris)

model = LinearSVC()
model.fit(X_train, y_train)

print("train score:" + str(model.score(X_train, y_train)) + "\n" +
      "test score:" + str(model.score(X_test, y_test)))
print()
print("label:", y_test[:10])
print("prediction:", model.predict(X_test[:10]))
