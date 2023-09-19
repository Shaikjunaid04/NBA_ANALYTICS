
import pandas as pd 
import numpy as np
import plotly.express as px 
import plotly.graph_objects as go 
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import matplotlib as mpl
from plotly.offline import iplot

pd.set_option('display.max_columns',None)

df = pd.read_csv('NBA_scrape_data_Final.csv')


########################
df.sample(10)
df.isna().sum()


########################

#Dropping the columns : RANK COLUMN IS A arbitary column its a meaning less column in stats and effeciency as well 
df.drop(columns=['Unnamed: 0','RANK'], inplace=True)

########################
#creating a new column that identifies year in a integer from : 

df['season_start_year']=df['Year'].str[:4].astype(int)

########################

df['TEAM'].replace(to_replace=['NOP','NOH'], value='NO', inplace=True)

########################

#cleaning up season_type column
df['Season_type'].replace('Regular%20Season', 'Regular',inplace=True)

########################
#preparing two data frames : 1-regular   2- playoffs  
rs_df=df[df['Season_type']=='Regular']
playoff_df=df[df['Season_type']=='Playoffs']

########################
df.columns 

########################


total_cols=['MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA',
       'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF',
       'PTS']

########################
df

########################

msno.bar(df)
msno.matrix(df)

#######################

df.to_csv('NBA_Cleaned_DATA.csv')

########################







