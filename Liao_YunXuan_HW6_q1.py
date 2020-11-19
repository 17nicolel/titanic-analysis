# YunXuan Liao
# ITP 449 Fall 2020
# HW6
# Q1

import matplotlib.pyplot as plt
import seaborn as sb
import sklearn
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report

titanic= pd.read_csv('Titanic.csv', header=0)
# read the dataset into a dataframe

print(titanic.head())
print(titanic.shape)
#target variable is survived
# Explore the dataset and determine what is the target variable.

titanic.drop(titanic.columns[[0]], axis=1, inplace=True)
print(titanic.head)
#Drop factor(s) that are not likely to be relevant for logistic regression.

print(titanic.isnull().sum())
#Make sure there are no missing values.

plt.figure(1)
sb.countplot(x='Survived', data=titanic)
plt.figure(2)
sb.countplot(x='Class', data=titanic)
plt.figure(3)
sb.countplot(x='Sex', data=titanic)
plt.figure(4)
sb.countplot(x='Age', data=titanic)
plt.show()
#Plot count plots of each of the remaining factors.

titanic2 = pd.get_dummies(titanic, columns=['Class', 'Sex', 'Age'])
print(titanic2.head())
#Convert all categorical variables into dummy variables

X = titanic2.iloc[:,1:]
y = titanic2.iloc[:,0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=2020)
#Partition the data into train and test sets (70/30). Use random_state = 2020

LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)
y_pred = LogReg.predict(X_test)
#Fit the training data to a logistic regression model

print(metrics.accuracy_score(y_test, y_pred))
#Display the accuracy of your predictions for survivability

confusionmatrix = metrics.confusion_matrix(y_test, y_pred)
print(confusionmatrix)
plot_confusion_matrix(LogReg, X_test, y_test)
plt.show()
#Display the confusion matrix along with the labels (Yes, No).

test = [[0,0,1,0,0,1,1,0]]
result = LogReg.predict(test)
print(result)
#Now, display the predicted value of the survivability of a male adult passenger traveling in 3rd class.
#he is predicted to die.