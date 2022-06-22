import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import confusion_matrix

data = pd.read_csv("https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/"
                   "raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv")
print(data.head())
X=data.iloc[:,:4]
Y=data.iloc[:,-1]
sns.heatmap(data.corr())
plt.title("Heat map")
plt.show()

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=.75,random_state=0)

svc_linear=SVC(kernel="linear", random_state=0)
svc_linear.fit(X_train,Y_train)
y_pred1=svc_linear.predict(X_test)
print("Confusion Matrix=\n",confusion_matrix(Y_test,y_pred1))
acc1=cross_val_score(estimator=svc_linear,X=X_train,y=Y_train,cv=10)
print("Accuracy:",acc1.mean()*100)
print("Standard Deviation",acc1.std()*100)

svc_poly=SVC(kernel="poly", random_state=0)
svc_poly.fit(X_train,Y_train)
y_pred2=svc_poly.predict(X_test)
print("Confusion Matrix=\n",confusion_matrix(Y_test,y_pred2))
acc2=cross_val_score(estimator=svc_poly,X=X_train,y=Y_train,cv=10)
print("Accuracy:",acc2.mean()*100)
print("Standard Deviation",acc2.std()*100)


svc_sig=SVC(kernel="sigmoid", random_state=0)
svc_sig.fit(X_train,Y_train)
y_pred3=svc_sig.predict(X_test)
print("Confusion Matrix=\n",confusion_matrix(Y_test,y_pred3))
acc3=cross_val_score(estimator=svc_sig,X=X_train,y=Y_train,cv=10)
print("Accuracy:",acc3.mean()*100)
print("Standard Deviation",acc3.std()*100)
