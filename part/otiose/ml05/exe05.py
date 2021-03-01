from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.svm import LinearSVC
from warnings import filterwarnings

filterwarnings('ignore')

fig, axes = plt.subplots(nrows=1, ncols=2)

data1 = pd.read_csv("data1.csv")[["age", "fare", "survived"]].dropna()
X_train, X_test, y_train, y_test = train_test_split(data1[["age", "fare"]],
                                                    data1[["survived"]])

model = LinearSVC()
model.fit(X_train, y_train.to_numpy().T[0])

axes[0].scatter(X_train.age, X_train.fare, c=y_train.survived)
axes[0].scatter(X_test.age, X_test.fare, c=model.predict(X_test), alpha=0.4)
axes[1].scatter(data1.age, data1.fare, c=data1.survived)

plt.show()
