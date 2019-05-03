# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 18:02:42 2019

@author: Zhou
"""

# Import Module
import pandas as pd 
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Read data and resplit dataset
train_data_set = pd.read_csv("Data/Train_Data_Set.csv")
test_data_set = pd.read_csv("Data/Test_Data_Set.csv")

X_train = train_data_set.iloc[:,:-1]
y_train = train_data_set.iloc[:,-1]

X_test = test_data_set.iloc[:,:-1]
y_test = test_data_set.iloc[:,-1]

# The whole dataset
data = pd.DataFrame(np.concatenate((train_data_set,test_data_set), axis =0))
data_feature = data.iloc[:,:-1]
data_target = data.iloc[:,-1]
# Resplit the dataset
new_X_train, new_X_test, new_y_train, new_y_test = train_test_split(data_feature , data_target, 
                                                                    test_size = 0.21) 

clf = GaussianNB()
clf.fit(new_X_train, new_y_train)
new_y_pred = clf.predict(new_X_test)

# Gaussian Naive Bayes
clf = GaussianNB()
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
accuracy_score(y_pred, y_test)

# Print accuracy score
print(accuracy_score(new_y_pred, new_y_test))
print(accuracy_score(y_pred,y_test))

print(confusion_matrix(new_y_test, new_y_pred))
# You can see that it has big difference with resplitted dataset and original set


