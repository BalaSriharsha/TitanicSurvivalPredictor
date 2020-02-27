#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 15:03:39 2020

@author: balasriharsha
"""
#Importing Modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing Dataset
dataset = pd.read_csv("/home/balasriharsha/Desktop/TitanicSurvivalPredictor/titanic.csv")

#Removing Unnecessary Columns
dataset.drop(["Name","Ticket"],axis=1,inplace=True)

#Dealing with Missing Values(One)
dataset['Cabin'] = dataset['Cabin'].fillna('C106')
dataset['Embarked'] = dataset['Embarked'].fillna('Q')

#Seperating Independent and Dependent Variables
X = dataset.iloc[:,2:].values
Y = dataset.iloc[:,1].values

#Dealing with missing variables(two)
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy='mean')
X[:,2:3]=imputer.fit_transform(X[:,2:3])

#Dealing with Categorical Data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le = LabelEncoder()
oe = OneHotEncoder(categorical_features=[1,-2,-1])
X[:,1]=le.fit_transform(X[:,1])
X[:,-2]=le.fit_transform(X[:,-2])
X[:,-1]=le.fit_transform(X[:,-1])

X = oe.fit_transform(X).toarray()

#Delaing with Dummy Variable Trap
X = np.delete(arr=X,obj=0,axis=1)
X = np.delete(arr=X,obj=1,axis=1)
X = np.delete(arr=X,obj=149,axis=1)

#Multiple Linear Regression
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,Y_train)

Y_pred_lr = lr.predict(X_test)

#Standard Scaling for SVR
Y = np.reshape(Y,(-1,1))

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)

from sklearn.model_selection import train_test_split
X_train_one,X_test_one,Y_train_one,Y_test_one = train_test_split(X,Y,test_size=0.2,random_state=0)

#SVR Regression
from sklearn.svm import SVR
sr = SVR()
sr.fit(X_train_one,Y_train_one)

Y_pred_sr = sc_Y.inverse_transform(sr.predict(X_test_one))

#Getting Ready for Decision Tree Regression
X = sc_X.inverse_transform(X)
Y = sc_Y.inverse_transform(Y)

from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(random_state=0)
dtr.fit(X_train,Y_train)

Y_pred_dtr = dtr.predict(X_test)

#Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=1000,random_state=0)
rfr.fit(X_train,Y_train)

Y_pred_rfr = rfr.predict(X_test)

#Changing Y_pred values to 0's and 1's
for i in range(len(Y_pred_sr)):
    if(Y_pred_sr[i]<0.5):
        Y_pred_sr[i]=0
    else:
        Y_pred_sr[i]=1

for i in range(len(Y_pred_lr)):
    if(Y_pred_lr[i]<0.5):
        Y_pred_lr[i]=0
    else:
        Y_pred_lr[i]=1
        
for i in range(len(Y_pred_rfr)):
    if(Y_pred_rfr[i]<0.5):
        Y_pred_rfr[i]=0
    else:
        Y_pred_rfr[i]=1

#Percentage of Accuracy for Each Model
sr_count = 0
lr_count = 0
dtr_count= 0
rfr_count= 0
for i in range(len(Y_test)):
    if(int(Y_pred_sr[i])==Y_test[i]):
        sr_count += 1
for i in range(len(Y_test)):
    if(int(Y_pred_lr[i])==Y_test[i]):
        lr_count += 1
for i in range(len(Y_test)):
    if(int(Y_pred_dtr[i])==Y_test[i]):
        dtr_count += 1
for i in range(len(Y_test)):
    if(int(Y_pred_rfr[i])==Y_test[i]):
        rfr_count += 1
print("Percentage of Accuracy for SVR is "+str((sr_count*100)/179))
print("Percentage of Accuracy for Multiple Linear Regression is "+str((lr_count*100)/179))
print("Percentage of Accuracy for Decision Tree Regression is "+str((dtr_count*100)/179))
print("Percentage of Accuracy for Random Forest Regression is "+str((rfr_count*100)/179))
