
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load NBA data (replace 'nba_data.csv' with your dataset file path)
nba_data = pd.read_csv('player_mvp_stats.csv')

# Filter data to include only MVP award winners
mvp_winners = nba_data[nba_data['Pts Won'] > 0]  # Assuming 'Pts Won' is the column for MVP points

# Count the number of MVPs for each team
mvp_counts = mvp_winners['Team'].value_counts()

# Create a bar plot to visualize the number of MVPs per team
plt.figure(figsize=(12, 8))
sns.barplot(x=mvp_counts.values, y=mvp_counts.index, palette='viridis')
plt.title('Number of MVPs per NBA Team')
plt.xlabel('Number of MVPs')
plt.ylabel('Team')
plt.show()

##################


# Create a scatter plot to visualize the relationship between FTA and PTS


plt.figure(figsize=(10, 6))
sns.scatterplot(x='FTA', y='PTS', data=nba_data, alpha=0.5, color='blue')
plt.title('Relation Between Free Throws Attempted (FTA) and Total Points (PTS)')
plt.xlabel('Free Throws Attempted (FTA)')
plt.ylabel('Total Points (PTS)')
plt.grid(True)
plt.show()


##################




# Creating a plot to visualize the distribution of FG%


plt.figure(figsize=(10, 6))
plt.hist(nba_data['FG%'], bins=20, rwidth=0.8, density=True, color='blue', alpha=0.7)
plt.title('Distribution of Field Goal Percentage (FG%)')
plt.xlabel('FG%')
plt.ylabel('Frequency')
plt.show()


##################

# Calculate the total points for each category - scoring type 



total_ft_points = nba_data['FT'].sum()
total_2p_points = nba_data['2P'].sum()
total_3p_points = nba_data['3P'].sum()

# Create a pie chart
labels = ['Free Throws (FT)', '2-Point Field Goals (2P)', '3-Point Field Goals (3P)']
sizes = [total_ft_points, total_2p_points, total_3p_points]
colors = ['gold', 'lightcoral', 'lightskyblue']
explode = (0.1, 0, 0)  # Explode the first slice (i.e., 'Free Throws')

plt.figure(figsize=(8, 8))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Distribution of Total Points by Scoring Type')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


##################
nba_data.columns
# a violin plot for PTS by player position

plt.figure(figsize=(12, 6))
sns.violinplot(data=nba_data, x='Pos', y='PTS', palette='Set2')
plt.title('Points Scored (PTS) by Player Position')
plt.xlabel('Player Position (Pos)')
plt.ylabel('Total Points Scored (PTS)')
plt.xticks(rotation=0)
plt.show()

##################


# a 2D density plot of Age vs. PTS
plt.figure(figsize=(10, 6))
sns.kdeplot(data=nba_data, x='Age', y='PTS', fill=True, cmap="Blues", thresh=0, levels=100)
plt.title('2D Density Plot of Age vs. Points (PTS)')
plt.xlabel('Age')
plt.ylabel('Points (PTS)')
plt.show()

##################





