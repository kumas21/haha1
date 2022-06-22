import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
data=pd.read_csv("https://raw.githubusercontent.com/Gokulgoky1/linear-regression-data-set/main/student_scores.csv")
print(data)
X=data.iloc[:,[0]]
y=data.iloc[:,[1]]
print(y)

plt.scatter(data['Hours'],data['Scores'],alpha=.5)
plt.title('Hours Studied Vs Score')
plt.xlabel('Hours Studied')
plt.ylabel('Scores Obtained')
plt.show()

X_train,X_test,Y_train,Y_test=train_test_split(X,y,train_size=.8,random_state=0)
regression1=LinearRegression()
regression1.fit(X_train,Y_train)

print("intercept",regression1.intercept_)
print("Coef",regression1.coef_)

y_pred=regression1.predict(X_test)

absolute_error=mean_absolute_error(Y_test,y_pred)
print("Mean Absolute error:",absolute_error)
mean_squared_err=mean_squared_error(Y_test,y_pred)
print("Mean Squared Error",mean_squared_err)

predicted_score=(regression1.coef_*X)+regression1.intercept_
plt.scatter(data['Hours'],data['Scores'],alpha=.5)
plt.plot(X,predicted_score)
plt.title('Hours Studied Vs Score')
plt.xlabel('Hours Studied')
plt.ylabel('Scores Obtained')
plt.show()