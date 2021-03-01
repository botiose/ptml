from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

data1 = pd.read_csv("data1.csv")[["age", "fare", "sex", "survived"]].dropna()
d = {"male": 1, 'female': 0}
data1.sex = data1.sex.map(d)

X_train, X_test, y_train, y_test = train_test_split(data1[["age", "fare",
                                                           "sex"]],
                                                    data1[["survived"]])

model = LogisticRegression()
model.fit(X_train, y_train.to_numpy().T[0])

print("train score:" + str(model.score(X_train, y_train)) + "\n" +
      "test score:" + str(model.score(X_test, y_test)))
print()
print("label:", y_test[:10].to_numpy().T[0])
print("prediction:", model.predict(X_test[:10]))
