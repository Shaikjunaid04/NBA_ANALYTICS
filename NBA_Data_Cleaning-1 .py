
#CLEANING player_mvp data 
import pandas as pd


mvps=pd.read_csv('mvps.csv')

#cleaning MVP vote dataset 
mvps=mvps[['Player','Year','Pts Won','Pts Max','Share']]
mvps.head()
players=pd.read_csv('players.csv')
players 



###################
#deleting some unwanted columnns
del players['Unnamed: 0']
del players['Rk']



###################
players.head()

#fixing up players name : 
players['Player'].head(50)


players['Player']=players['Player'].str.replace("*","",regex=False)


###################

players.head(50)




#make sure each player has only one row : so we can predict how many votes each player will get 

players.groupby(["Player","Year"])

def single_row(df):
    if df.shape[0]==1:
        return df
    else: 
        row=df[df['Tm']=='TOT']
        row['Tm']=df.iloc[-1,:]["Tm"]
        return row
    
players = players.groupby(["Player","Year"]).apply(single_row)
players.head(20)

players.index =players.index.droplevel()
players.index =players.index.droplevel()
players


###################

#combining player and mvp data : 
combined= players.merge(mvps,how='outer',on=['Player','Year'])
combined
combined[["Pts Won","Pts Max","Share"]]=combined[["Pts Won","Pts Max","Share"]].fillna(0)





###################


#Cleaning team data 

teams=pd.read_csv('teams.csv')
teams

#removing division: 
teams = teams[~teams["W"].str.contains("Division")]

teams["Team"] = teams["Team"].str.replace("*","",regex=False)

teams.head(5)

teams["Team"].unique()
combined["Tm"].unique()
 ###################


nicknames = {}
with open("nicknames.txt") as f:
    lines = f.readlines()
    for line in lines[1:]:
        abbrev,name = line.replace("\n","").split(",")
        nicknames[abbrev] = name



combined["Team"] = combined["Tm"].map(nicknames)

combined.head()
train = combined.merge(teams, how="outer",on=["Team", "Year"])

train

del train["Unnamed: 0"]
train
train.dtypes 


train = train.apply(pd.to_numeric, errors='ignore')
train.dtypes

train["GB"].unique()
train["GB"] = pd.to_numeric(train["GB"].str.replace("—","0"))
train.dtypes




###################  

# Define the combinations to remove
# Define the combinations to remove
combinations_to_remove = ['SF-SG', 'PG-SG', 'SG-SF', 'C-PF', 'SG-PG', 'PF-C', 'PF-SF', 'SF-PF', 'SG-PF', 'PG-SF', 'SF-C']

# Strip whitespace from the 'Pos' column
train['Pos'] = train['Pos'].str.strip()

# Filter the DataFrame to exclude rows with the specified combinations
filtered_df = train[~train['Pos'].isin(combinations_to_remove)]


filtered_df.to_csv("player_mvp_stats.csv")



filtered_df['Pos'].value_counts()


highest_scoring = train[train["G"] > 70].sort_values("PTS", ascending=False).head(10)


highest_scoring.plot.bar("Player", "PTS")


highest_scoring_by_year = train.groupby("Year").apply(lambda x: x.sort_values("PTS", ascending=False).head(1))


highest_scoring_by_year.plot.bar("Year", "PTS")

train.groupby("Year").apply(lambda x: x.shape[0])


train.corr()["Share"]

train.corr()["Share"].plot.bar()
filtered_df





