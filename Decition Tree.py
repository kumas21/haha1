import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.tree import plot_tree,export_text
data=pd.read_csv("https://raw.githubusercontent.com/Bijay555/Decision-Tree-Classifier/master/iris.csv")
print(data)
print(data.describe())
X=data.iloc[:,:4]
Y=data.iloc[:,-1]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=.7,random_state=0)

DTC=DecisionTreeClassifier(criterion='gini')
DTC.fit(X_train,Y_train)
print('Score',DTC.score(X_test,Y_test))
y_pred=DTC.predict(X_test)
cm=confusion_matrix(Y_test,y_pred)
acc=accuracy_score(Y_test,y_pred)
print("confusion Matrix\n",cm)
print("Accuracy Score",acc)
target=list(data['Species'].unique())
feature_names=list(X.columns)

plot_tree(DTC,feature_names=feature_names,class_names=target,filled=True,rounded=True)
plt.show()
r=export_text(DTC,feature_names=feature_names)
print(r)







