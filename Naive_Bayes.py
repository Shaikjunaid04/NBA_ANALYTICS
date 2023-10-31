#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:11:55 2023

@author: shaikjunaid
"""

from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load the dataset
data = pd.read_csv('NBA_Team_Stats.csv')
# Checking for missing values
print(data.isnull().sum())

# Removing rows with missing values in any column
data = data.dropna()
# Assuming the 'Year' column format is '1990-1991', modifying it to '1990'
data['Year'] = data['Year'].apply(lambda x: x.split('-')[0])





# Specify the numeric attributes for normalization
numeric_attributes = ['G', 'Min', 'Reb', 'Ast', 'Stl', 'Blk', 'To', 'Pf', 'Dreb', 'Oreb', 'Fgm-a', 'Pct', '3gm-a', 'Pct', 'Ftm-a', 'Pct', 'Eff', 'Deff']

# Filter the DataFrame to include only numeric attributes
numeric_data = data[numeric_attributes]

# Normalizing the numeric data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(numeric_data)

# Creating a DataFrame from the scaled data
normalized_df = pd.DataFrame(scaled_data, columns=numeric_attributes)

# Display the first few rows of the normalized DataFrame
print(normalized_df.head())

normalized_df


categorical_features = set(["Year"])

for cat in categorical_features:
    data[cat] = data[cat].astype("category")

data.info()

print("Unique classes for each categorical feature:")
for cat in categorical_features:
    print("{:15s}".format(cat), "\t", len(data[cat].unique()))
    
    
    
onehot_cols = categorical_features

data1 = pd.get_dummies(data, columns=onehot_cols, drop_first=True)


# Features you want to use for prediction (adjust as needed)
features = ['G', 'Min', 'Reb', 'Ast', 'Stl', 'Blk', 'To', 'Pf', 'Dreb', 'Oreb', 'Fgm-a', 'Pct', '3gm-a', 'Pct', 'Ftm-a', 'Pct', 'Eff', 'Deff']

# Splitting the data into features and target variable
X_sample = data1[features]
y_sample = data1['Playoffs']

X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# Create a Naive Bayes classifier and fit the model
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = gnb.predict(X_test)
y_pred_train=gnb.predict(X_train)

# Calculate the accuracy score of the model
accuracy_train= accuracy_score(y_train, y_pred_train)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Accuracy_Train: {:.2f}%".format(accuracy_train * 100))

confmat=confusion_matrix(y_test, y_pred)

index = ["Actual Not in Playoffs", "Actual Playoffs"]
columns = ["Predicted Not in playoffs", "Predicted Actual playoffs"]
conf_matrix = pd.DataFrame(data=confmat, columns=columns, index=index)
plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Confusion Matrix - Naive_Bayes")
plt.show()



print(classification_report(y_test, y_pred))
