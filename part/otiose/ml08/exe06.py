from sklearn import svm
from sklearn.metrics import roc_curve
import pandas as pd
from matplotlib import pyplot as plt

data1 = pd.read_csv("data1.csv")[["age", "fare", "survived"]].dropna()

model = svm.SVC(probability=True)
model.fit(data1[["age", "fare"]], data1.survived)
fpr, tpr, _ = roc_curve(data1.survived,
                        model.predict_proba(data1[["age", "fare"]])[:,1])
plt.plot(fpr, tpr, "r")
plt.show()


