#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 18:52:08 2023

@author: shaikjunaid
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
import plotly.graph_objects as go
import matplotlib as mpl
from plotly.offline import iplot
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import  plot_tree
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the dataset (assuming the dataset is in a file named 'basketball_data.csv')
df = pd.read_csv('CLEANED_FINAL_DATA_1.csv')

​
# Checking for missing values
print(data.isnull().sum())
​
# Removing rows with missing values in any column
data = data.dropna()

# Assuming the 'Year' column format is '1990-1991', modifying it to '1990'
data['Year'] = data['Year'].apply(lambda x: x.split('-')[0])


print(data.dtypes)
# Checking data types of columns


def convert_to_float(value):
    if '-' in value:
        parts = value.split('-')
        return (float(parts[0]) + float(parts[1])) / 2
    else:
        return float(value)  # Convert simple values to float

data['Fgm-a'] = data['Fgm-a'].apply(lambda x: convert_to_float(x))
data['3gm-a'] = data['3gm-a'].apply(lambda x: convert_to_float(x))
data['Ftm-a'] = data['Ftm-a'].apply(lambda x: convert_to_float(x))
​

print(data.dtypes)

# Lists of teams in the East and West conferences
east_conference_teams = ["Miami", "Detroit", "Cleveland", "Boston", "Chicago", "Indiana", "Orlando", "New Jersey",
                         "Philadelphia", "New York", "Toronto", "Washington", "Milwaukee", "Charlotte"]

west_conference_teams = ["San Antonio", "Phoenix", "Dallas", "Minnesota", "Houston", "Denver", "Sacramento",
                         "L.A.Clippers", "Memphis", "Seattle", "Utah", "Portland", "Golden State", "L.A.Lakers",
                         "New Orleans"]

# Creating a new column 'Conference' based on team lists
data['Conference'] = data['Team'].apply(lambda x: 'East' if x in east_conference_teams else 'West')
​

data.head(20)


corr_matrix = data.corr()

plt.figure(figsize=(15, 10))
sns.heatmap(corr_matrix, vmin=-1, vmax=1, cmap="seismic")
plt.gca().patch.set(hatch="df1", edgecolor="#666")
plt.show()


# Thresholds for qualification
threshold = {
    'Pts': 95,
    'G': 82
}

# Determine if a team qualifies for playoffs based on the threshold
data['Playoffs'] = data.apply(lambda x: 'IN playoffs' if x['Pts'] > threshold['Pts'] and x['G'] > threshold['G'] else 'Not in playoffs', axis=1)


Playoffs_counts = data['Playoffs'].value_counts()
​
# Display the count of unique positions
print(Playoffs_counts)
​
plt.figure(figsize=(10, 8))
plt.title("Histogram for the Playoffs_counts")
sns.barplot(x= Playoffs_counts.index, y= Playoffs_counts.values)
plt.xlabel("Playoffs")
plt.ylabel("Value")
plt.show()


# Specify the numeric attributes for normalization
numeric_attributes = ['G', 'Min', 'Reb', 'Ast', 'Stl', 'Blk', 'To', 'Pf', 'Dreb', 'Oreb', 'Fgm-a', 'Pct', '3gm-a', 'Pct', 'Ftm-a', 'Pct', 'Eff', 'Deff']
​
# Filter the DataFrame to include only numeric attributes
numeric_data = data[numeric_attributes]
​
# Normalizing the numeric data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(numeric_data)
​
# Creating a DataFrame from the scaled data
normalized_df = pd.DataFrame(scaled_data, columns=numeric_attributes)
​
# Display the first few rows of the normalized DataFrame
print(normalized_df.head())
​

normalized_df