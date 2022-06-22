import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score

data = pd.read_csv("https://raw.githubusercontent.com/reshma78611/Classification-using-Naive-Bayes-with-Python/main/Diabetes_RF.csv")

X=data.iloc[:,:8]
Y=data.iloc[:,8]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=.7, random_state=0)
GBmodel=GaussianNB()
GBmodel.fit(X_train,Y_train)
Y_pred=GBmodel.predict(X_test)
cm=confusion_matrix(Y_test,Y_pred)
acc=accuracy_score(Y_test,Y_pred)
print("Confusion Matrix:\n",cm)
print("Accuracy_score:\n",acc)
