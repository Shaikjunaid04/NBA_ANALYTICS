import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("nba_games.csv", index_col=0)


main_features = [ 'fg%', '3p%', 'ft%', 
                          'orb', 'drb', 'orb%', 'drb%', 'ft%_max', 'fg%_max', '3p%_max', 
                          'fg%_opp', '3p%_opp', 'ft%_opp', 'orb_opp', 'trb_opp', 'orb%_opp', 
                          'drb%_opp', 'ft%_max_opp', 'fg%_max_opp', '3p%_max_opp' ,'won']  

# Creating a correlation heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(df[main_features].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Main Features')
plt.show()

import missingno as msno
import matplotlib.pyplot as plt


# Visualizing missing values 
fig, ax = plt.subplots(figsize=(10, 6))
msno.matrix(df, color=(0.25, 0.4, 0.8), ax=ax)

# Adjust the layout to minimize gaps
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

plt.title('Missing Values Visualization', fontsize=16)
plt.xlabel('Features', fontsize=12)
plt.ylabel('Data Points', fontsize=12)
plt.show()


df = pd.DataFrame(df)

# Check for null values
null_values = df.isnull().sum()
print("Null values in each column:")
print(null_values)

df = df.sort_values("date")

df = df.reset_index(drop=True)

del df["mp.1"]
del df["mp_opp.1"]
del df["index_opp"]

def add_target(group):
    group["target"] = group["won"].shift(-1)
    return group

df = df.groupby("team", group_keys=False).apply(add_target)

df["target"][pd.isnull(df["target"])] = 2
df["target"] = df["target"].astype(int, errors="ignore")

df["won"].value_counts()

won_counts = df["won"].value_counts()



# Plotting
plt.figure(figsize=(8, 6))
sns.barplot(x=won_counts.index, y=won_counts.values, palette="viridis")
plt.title("Distribution of Game Outcomes")
plt.xlabel("Game Outcome")
plt.ylabel("Count")
plt.xticks([0, 1], ['Lost', 'Won'])  # Assuming 0 is Lost and 1 is Won
plt.show()

nulls = pd.isnull(df).sum()

nulls = nulls[nulls > 0]

valid_columns = df.columns[~df.columns.isin(nulls.index)]

df = df[valid_columns].copy()