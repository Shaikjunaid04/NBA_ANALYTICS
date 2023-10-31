#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 19:06:35 2023

@author: shaikjunaid
"""

rom sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


# Features you want to use for prediction (adjust as needed)
features = ['Blk', 'To', 'Pf', 'Dreb', 'Oreb', 'Fgm-a', 'Pct', '3gm-a', 'Pct', 'Ftm-a', 'Pct', 'Eff', 'Deff']

# Splitting the data into features and target variable
X = data[features]
y = data['Playoffs']

# Normalizing the features
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)
X_normalized = pd.DataFrame(X_normalized, columns=features)

​
# Splitting the normalized data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)
​
# Initialize Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
​
# Train the model
clf.fit(X_train, y_train)
​
# Predict using the test set
predictions = clf.predict(X_test)
​
# Confusion matrix
cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(cm)
​


dtc = DecisionTreeClassifier(random_state=42)
parameters = [{"criterion": ["gini"], "max_depth": [5, 10,15], "min_samples_split": [5000,1000]}]
grid = GridSearchCV(dtc, parameters, verbose=5, n_jobs=-1)
grid.fit(X_train, y_train)
#dtc=grid.best_params_
​
​
print("Best parameters scores:")
print(grid.best_params_)
print("Train score:", grid.score(X_train, y_train))
print("Validation score:", grid.score(X_test, y_test))
​

​
print("Default scores:")
dtc.fit(X_train, y_train)
print("Train score:", dtc.score(X_train, y_train))
print("Validation score:", dtc.score(X_test, y_test))
​
y_pred = dtc.predict(X_test)
​
accuracy_score(y_test, y_pred)
f1_score(y_test, y_pred, average="macro")
​
print(classification_report(y_test, y_pred))
​
accuracy_score(y_test, y_pred)
​
y_pred = dtc.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
​
index = ["Actual Not in Playoffs", "Actual Playoffs"]
columns = ["Predicted Not in playoffs", "Predicted Actual playoffs"]
conf_matrix = pd.DataFrame(data=confmat, columns=columns, index=index)
plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Confusion Matrix - Decision Tree")
plt.show()

#important features
importances = pd.DataFrame(np.zeros((X_train.shape[1], 1)), columns=["importance"], index=X_train.columns)
​
importances.iloc[:,0] = dtc.feature_importances_
​
importances = importances.sort_values(by="importance", ascending=False)[:30]
​
plt.figure(figsize=(15, 10))
sns.barplot(x="importance", y=importances.index, data=importances)
plt.show()

#plotting the tree
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(dtc, max_depth=4, fontsize=10, feature_names=X_train.columns.to_list(), class_names = True, filled=True)
plt.show()
​
# create dot file with max depth of 3
dot_data = export_graphviz(dtc, out_file=None, feature_names=X_train.columns, class_names=['Not in Playoffs', 'Playoffs'], filled=True, rounded=True, special_characters=True, max_depth=5)
​
# create graph from dot file
graph = graphviz.Source(dot_data)
​
# show graph
graph.view()



dtc1 = DecisionTreeClassifier(random_state=42, criterion = 'entropy', max_depth = 100, min_samples_split= 10)
​
print("Default scores:")
dtc1.fit(X_train, y_train)
print("Train score:", dtc1.score(X_train, y_train))
print("Validation score:", dtc1.score(X_test, y_test))
​
# Make predictions on the testing set
y_pred = dtc1.predict(X_test)
y_pred_train=dtc1.predict(X_train)
​
# Calculate the accuracy score of the model
accuracy_train= accuracy_score(y_train, y_pred_train)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Accuracy_Train: {:.2f}%".format(accuracy_train * 100))
​
y_pred = dtc1.predict(X_test)
​
accuracy_score(y_test, y_pred)
f1_score(y_test, y_pred, average="macro")
​
print(classification_report(y_test, y_pred))
​
y_pred = dtc1.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
​
​
index = ["Actual Not in Playoffs", "Actual Playoffs"]
columns = ["Predicted Not in playoffs", "Predicted Actual playoffs"]
conf_matrix = pd.DataFrame(data=confmat, columns=columns, index=index)
plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Confusion Matrix - Decision Tree")
plt.show()
​
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(dtc1, max_depth=5, fontsize=10, feature_names=X_train.columns.to_list(), class_names = True, filled=True)
plt.show()
​
# create dot file with max depth of 3
dot_data = export_graphviz(dtc1, out_file=None, feature_names=X_train.columns, class_names=['Not Severe', 'Severe'], filled=True, rounded=True, special_characters=True, max_depth=7)
​
# create graph from dot file
graph = graphviz.Source(dot_data)
​
# show graph
graph.view()




dtc2 = DecisionTreeClassifier(random_state=42, criterion = 'gini', max_depth = 10, min_samples_split= 100)
​
​
print("Default scores:")
dtc2.fit(X_train, y_train)
print("Train score:", dtc2.score(X_train, y_train))
print("Validation score:", dtc2.score(X_test, y_test))
​
# Make predictions on the testing set
y_pred = dtc2.predict(X_test)
y_pred_train=dtc2.predict(X_train)
​
# Calculate the accuracy score of the model
accuracy_train= accuracy_score(y_train, y_pred_train)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Accuracy_Train: {:.2f}%".format(accuracy_train * 100))
​
y_pred = dtc2.predict(X_test)
​
accuracy_score(y_test, y_pred)
f1_score(y_test, y_pred, average="macro")
​
print(classification_report(y_test, y_pred))
​
y_pred = dtc2.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
​
index = ["Actual Not in Playoffs", "Actual Playoffs"]
columns = ["Predicted Not in playoffs", "Predicted Actual playoffs"]
conf_matrix = pd.DataFrame(data=confmat, columns=columns, index=index)
plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Confusion Matrix - Decision Tree")
plt.show()
​
​
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(dtc2, max_depth=4, fontsize=10, feature_names=X_train.columns.to_list(), class_names = True, filled=True)
plt.show()
​
​
# create dot file with max depth of 3
dot_data = export_graphviz(dtc2, out_file=None, feature_names=X_train.columns, class_names=['Not Severe', 'Severe'], filled=True, rounded=True, special_characters=True, max_depth=4)
​
# create graph from dot file
graph = graphviz.Source(dot_data)
​
# show graph
graph.view()





