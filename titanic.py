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
dataset = pd.read_csv("/home/balasriharsha/Desktop/ML/train.csv")

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

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,Y_train)

Y_pred_lr = lr.predict(X_test)

Y = np.reshape(Y,(-1,1))

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)

from sklearn.model_selection import train_test_split
X_train_one,X_test_one,Y_train_one,Y_test_one = train_test_split(X,Y,test_size=0.2,random_state=0)

from sklearn.svm import SVR
sr = SVR()
sr.fit(X_train_one,Y_train_one)

Y_pred_sr = sc_Y.inverse_transform(sr.predict(X_test_one))