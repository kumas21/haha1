import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

df=pd.read_csv("https://raw.githubusercontent.com/Gokulgoky1/linear-regression-data-set/main/diabetes.csv")
print(df)

X=df.drop(columns=["Outcome"])
Y=df.iloc[:,-1]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.3,stratify=Y)

# KNN
knn=KNeighborsClassifier()
knn_params={'n_neighbors':np.arange(1,25)}
knn_gs=GridSearchCV(knn,knn_params,cv=5)
knn_gs.fit(X_train,Y_train)
knn_best=knn_gs.best_estimator_
print("KNN Best Params:",knn_gs.best_params_)

# Random Forest
rf=RandomForestClassifier()
rf_params={'n_estimators':[50,100,200]}
rf_gs=GridSearchCV(rf,rf_params,cv=5)
rf_gs.fit(X_train,Y_train)
rf_best=rf_gs.best_estimator_
print("RF Best Params:",rf_gs.best_params_)

# Logistic Regression
lr=LogisticRegression()
lr.fit(X_train,Y_train)

# Calculating Scores
knn_score=knn_best.score(X_test,Y_test)
rf_score=rf_best.score(X_test,Y_test)
lr_score=lr.score(X_test,Y_test)
print("KNN:",knn_score)
print("RF:",rf_score)
print("Logistic Regression",lr_score)

# Average
Avg=(knn_score+lr_score+rf_score)/3
print("Avg=",Avg)

# Voting
# Create dictionary of Our Models
estimators=[('knn',knn_best),('rf',rf_best),('lr',lr)]
# Create our voting classifier, inputting our models
vt=VotingClassifier(estimators,voting='hard')
vt.fit(X_train,Y_train)
vt_score=vt.score(X_test,Y_test)
print("Voting Classifier Score=",vt_score)

# Bagging
bagging=BaggingClassifier()
bagging.fit(X_train,Y_train)
bagging_score=bagging.score(X_test,Y_test)
print("Bagging Score=",bagging_score)

# Boosting
boosting=AdaBoostClassifier()
boosting.fit(X_train,Y_train)
boosting_score=boosting.score(X_test,Y_test)
print("Boosting Score=",boosting_score)






