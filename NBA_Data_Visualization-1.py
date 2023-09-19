


import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os
from IPython.display import display, Image
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)

data = pd.read_csv('NBA_Cleaned_DATA.csv')

##################
data.shape
data.columns
total_cols = ['MIN','FGM','FGA','FG3M','FG3A','FTM','FTA',
              'OREB','DREB','REB','AST','STL','BLK','TOV','PF','PTS']



#heat  map 
data_per_min = data.groupby(['PLAYER','PLAYER_ID','Year'])[total_cols].sum().reset_index()
for col in data_per_min.columns[4:]:
    data_per_min[col] = data_per_min[col]/data_per_min['MIN']

data_per_min['FG%'] = data_per_min['FGM']/data_per_min['FGA']
data_per_min['3PT%'] = data_per_min['FG3M']/data_per_min['FG3A']
data_per_min['FT%'] = data_per_min['FTM']/data_per_min['FTA']
data_per_min['FG3A%'] = data_per_min['FG3A']/data_per_min['FGA']
data_per_min['PTS/FGA'] = data_per_min['PTS']/data_per_min['FGA']
data_per_min['FG3M/FGM'] = data_per_min['FG3M']/data_per_min['FGM']
data_per_min['FTA/FGA'] = data_per_min['FTA']/data_per_min['FGA']
data_per_min['TRU%'] = 0.5*data_per_min['PTS']/(data_per_min['FGA']+0.475*data_per_min['FTA'])
data_per_min['AST_TOV'] = data_per_min['AST']/data_per_min['TOV']

data_per_min = data_per_min[data_per_min['MIN']>=50]
data_per_min.drop(columns='PLAYER_ID', inplace=True)

fig = px.imshow(data_per_min.corr())
fig.show()


##################




# game trend - line plot
change_df = data.groupby('season_start_year')[total_cols].sum().reset_index()
change_df['POSS_est'] = change_df['FGA']-change_df['OREB']+change_df['TOV']+0.44*change_df['FTA']
change_df = change_df[list(change_df.columns[0:2])+['POSS_est']+list(change_df.columns[2:-1])]

change_df['FG%'] = change_df['FGM']/change_df['FGA']
change_df['3PT%'] = change_df['FG3M']/change_df['FG3A']
change_df['FT%'] = change_df['FTM']/change_df['FTA']
change_df['AST%'] = change_df['AST']/change_df['FGM']
change_df['FG3A%'] = change_df['FG3A']/change_df['FGA']
change_df['PTS/FGA'] = change_df['PTS']/change_df['FGA']
change_df['FG3M/FGM'] = change_df['FG3M']/change_df['FGM']
change_df['FTA/FGA'] = change_df['FTA']/change_df['FGA']
change_df['TRU%'] = 0.5*change_df['PTS']/(change_df['FGA']+0.475*change_df['FTA'])
change_df['AST_TOV'] = change_df['AST']/change_df['TOV']

change_df

##################


change_per48_df = change_df.copy()
for col in change_per48_df.columns[2:18]:
    change_per48_df[col] = (change_per48_df[col]/change_per48_df['MIN'])*48*5

change_per48_df.drop(columns='MIN', inplace=True)

fig = go.Figure()
for col in change_per48_df.columns[1:]:
    fig.add_trace(go.Scatter(x=change_per48_df['season_start_year'],
                             y=change_per48_df[col], name=col))
fig.show()


##################



del data['Unnamed: 0']


import os 
# Load the list of image files that ends with .png
image_files = [f for f in os.listdir('/Users/shaikjunaid/Desktop/img') if f.endswith('.png')]

# Extract player IDs from filenames (remove '.png')
PLAYER_ID = [f.replace('.png', '') for f in image_files]

# Create a DataFrame using the filenames and player IDs
data_1 = {'Filename': [os.path.join('/Users/shaikjunaid/Desktop/img', f) for f in image_files], 'PLAYER_ID': PLAYER_ID}

df_I = pd.DataFrame(data_1)

print(df_I)



for i in range(10):
    print(image_files[i])

df_I.head()['Filename'].values
df_I['PLAYER_ID'] = df_I['PLAYER_ID'].astype(int)
df_I.head()['PLAYER_ID'].values



# Display the sample top 3 images
for filename in df_I['Filename'].head():
    display(Image(filename=filename))

data = data.merge(df_I, left_on='PLAYER_ID', right_on='PLAYER_ID')  


def get_player_details(player_name):
    player_details = data[(data['PLAYER'] == player_name)]

    if not  player_details.empty:
        # Extract the row of company data
        player_row = player_details.iloc[0]

        # Display company details
        print(f"Player Name: {player_row['PLAYER']}")
        print(f"stats for the year : {player_row['season_start_year']}")
        print(f"Total points: {player_row['PTS']}")
        print(f"Field Goals Attempted : {player_row['FGA']}")
        print(f"Steals: {player_row['STL']}")
        print(f"Assists: {player_row['AST']}")
        print(f"Team: {player_row['TEAM']}")
        print(f"Field Goals Made : {player_row['FGM']}")
        print(f"Rebounds: {player_row['REB']}")
        
        
        # Display player image
        display(Image(filename=player_row['Filename']))
    else:
        print(f"No details found for company : {company}")

# Test the function with a specific company
get_player_details('Giannis Antetokounmpo')





def get_player_details(player_name):
    player_details = data[(data['PLAYER'] == player_name)]

    if not  player_details.empty:
        # Extract the row of company data
        player_row = player_details.iloc[0]

        # Display company details
        print(f"Player Name: {player_row['PLAYER']}")
        print(f"stats for the year : {player_row['season_start_year']}")
        print(f"Total points: {player_row['PTS']}")
        print(f"Field Goals Attempted : {player_row['FGA']}")
        print(f"Steals: {player_row['STL']}")
        print(f"Assists: {player_row['AST']}")
        print(f"Team: {player_row['TEAM']}")
        print(f"Field Goals Made : {player_row['FGM']}")
        print(f"Rebounds: {player_row['REB']}")
        
        
        # Display player image
        display(Image(filename=player_row['Filename']))
    else:
        print(f"No details found for company : {company}")

# Test the function with a specific company
get_player_details('Kevin Durant')


##################



specific_year = 2016
filtered_data = data[data['season_start_year'] == specific_year]


top_20_players = filtered_data.sort_values(by='STL', ascending=False).head(20)


fig = px.scatter(top_20_players, x='PLAYER', y='STL', size='STL', color='STL',
                 title=f'Top 20 Players with Highest Steals in {specific_year}',
                 labels={'PLAYER': 'Player', 'STL': 'STL'},
                 size_max=50)  # Adjust size_max to control circle size

# Customize the plot layout
fig.update_layout(showlegend=False)
fig.show()


##################





