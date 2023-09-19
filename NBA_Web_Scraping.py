#!/usr/bin/env python
# coding: utf-8

# In[84]:


import pandas as pd
import requests 
import time 
import numpy as np
pd.set_option('display.max_columns', None)



#storing the url in test_url for future use . 
test_url = 'https://stats.nba.com/stats/leagueLeaders?LeagueID=00&PerMode=Totals&Scope=S&Season=2012-13&SeasonType=Regular%20Season&StatCategory=PTS'


#data into json 
r= requests.get(url=test_url).json()


#headers of our data stored in var : table_headers . 
table_headers = r['resultSet']['headers']

#checking the rowset 

r['resultSet']['rowSet']

#for the first row : 

r['resultSet']['rowSet'][0]



#into data frame :"

pd.DataFrame(r['resultSet']['rowSet'], columns = table_headers)



#manipu;lating the URL to stack the data over n over 

temp_var_df1= pd.DataFrame(r['resultSet']['rowSet'], columns = table_headers)
temp_var_df2=pd.DataFrame({'Year':['2022-23'for i in range(len(temp_var_df1))],
                          'Season_type':['Regular%20Season'for i in range(len(temp_var_df1))]})
temp_var_df3= pd.concat([temp_var_df2,temp_var_df1],axis=1)
temp_var_df3



del temp_var_df1, temp_var_df2, temp_var_df3



df_cols = ['Year', 'Season_type']+ table_headers


pd.DataFrame(columns=df_cols)


headers = { 
     'Accept': '*/*',
     'Accept-Encoding':'gzip, deflate, br',
     'Accept-Language':'en-US,en;q=0.9',
     'Connection': 'keep-alive',
     'Host':'stats.nba.com',
     'Origin': 'https://www.nba.com',
     'Referer':'https://www.nba.com/',
     'Sec-Ch-Ua':'"Chromium";v="116", "Not)A;Brand";v="24", "Google Chrome";v="116"',
     'Sec-Ch-Ua-Mobile':'?0',
     'Sec-Ch-Ua-Platform':'"macOS"',
     'Sec-Fetch-Dest':'empty',
     'Sec-Fetch-Mode':'corsSec-Fetch-Site:same-site',
     'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36'

 }



df=pd.DataFrame(columns=df_cols)
Season_types=['Regular%20Season','Playoffs']
Years = ['2012-13','2013-14','2014-15','2015-16','2016-17','2017-18','2018-19','2019-20','2020-21','2021-22']

begin_loop= time.time()

for y in Years: 
    for s in Season_types:
        API_url='https://stats.nba.com/stats/leagueLeaders?LeagueID=00&PerMode=Totals&Scope=S&Season='+y+'&SeasonType='+s+'&StatCategory=PTS'
        r=requests.get(url=API_url).json()
        temp_var_df1= pd.DataFrame(r['resultSet']['rowSet'],columns = table_headers)
        temp_var_df2=pd.DataFrame({'Year':[y for i in range(len(temp_var_df1))],
                          'Season_type':[s for i in range(len(temp_var_df1))]})
        temp_var_df3= pd.concat([temp_var_df2,temp_var_df1],axis=1)
        df=pd.concat([df,temp_var_df3],axis=0)
        print(f'Finished scraping for the {y} {s}.')
        lag=np.random.uniform(low=5,high=40)
        print(f'...Waiting {round(lag,1)} seconds')
        time.sleep(lag)


print(f'process completed! Total run time : {((time.time()-begin_loop)/60,2)}')
df.to_csv('NBA_scrape_data.csv',index=False)
df

df.to_csv(r'/Users/shaikjunaid/Downloads/CLASS2023/ML/PROJECT /NBA_scrape_data_Final.csv')
