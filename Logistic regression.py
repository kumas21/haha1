import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score
data = pd.read_csv('https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/Social_Network_Ads'
                   '.csv')
print(data)
X= data.iloc[:, [2, 3]].values
y = data.iloc[:, [4]].values

X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=.25,random_state=0)

std_sc=StandardScaler()
X_train=std_sc.fit_transform(X_train)
X_test=std_sc.transform(X_test)

Loge_Classifier=LogisticRegression(random_state=0)
Loge_Classifier.fit(X_train,Y_train)
y_pred=Loge_Classifier.predict(X_test)
cm=confusion_matrix(Y_test,y_pred)
print('Confusion Matrix:','\n',cm)
acc=accuracy_score(Y_test,y_pred)
print('Accuracy Score:',acc)