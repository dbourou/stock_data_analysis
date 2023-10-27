#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 13:14:53 2023

@author: Dimitra Bourou
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 10:42:39 2023

@author: Dimitra Bourou
"""

import bs4 as bs # to crawl data from webpages
import pickle # to save python variables
import requests
import datetime as dt
import os
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
import numpy as np
import pickle
import matplotlib
matplotlib.use('QtAgg')
import sys
import pylab
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')


import matplotlib
matplotlib.use('QtAgg')
from matplotlib import style

import yahoo_fin.stock_info as si
import yahoo_fin.news as news
# the package is not very well maintained, last update 2021 and now tons of errors
# later susbstitute with another one, there are alternatives

from yahooquery import Ticker


style.use('ggplot')




# get some common lists of stocks, get also company data
# we can check if companies which correlate are always from same sectors
dow = si.tickers_dow()
nasdaq = si.tickers_nasdaq()
other = si.tickers_other(include_company_data=True) # but idk which stockexchange they are from, get company info

sp500 = si.tickers_sp500()

mystocks = ['biib', 'regn', 'aixa', 'una', 'meta', 'baba', 'vnm', 'ech' ]

tobuy = ['ewm','thd', 'givn' ]
# malaysian etf, thai etf, givadaun


# tickers = list(other['']
 # dow / nasdaq / other / sp500 / mystocks / tobuy

#tickers = tickers[0:100]


#columns = ['ACT Symbol', 'Security Name', 'Exchange', 'CQS Symbol', 'ETF',
#       'Round Lot Size', 'Test Issue', 'NASDAQ Symbol']

a1 = other['ACT Symbol'] == other['NASDAQ Symbol']
a2 = other['ACT Symbol'] == other['CQS Symbol']
a3 = other['CQS Symbol'] == other['NASDAQ Symbol']

a11 = a1.value_counts()
a21 = a2.value_counts()
a31 = a3.value_counts()

# there are many tickers... which one to use?
#tickers = list(other['']
# for simplicity just keep those that have the same ticker in all 3, they are the majority

other = other.loc[(a1 & a2 & a3)] 
other = other[['ACT Symbol', 'Security Name']]
other.rename(columns={'ACT Symbol': 'Ticker'}, inplace=True)

tickers = list(other['Ticker'].unique())

# find sector info with some other command

start = dt.datetime(2015,1,1)
end = dt.datetime(2023,1,1)
   

for ticker in tickers: # take all tickers later, will take a few mins
    print(ticker)    
    if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
        df = yf.download(ticker, start=start, end=end)
        df.to_csv('stock_dfs/{}.csv'.format(ticker))
    else:
        print('Already have {}'.format(ticker))

#maybe let it skip a ticker if its empty or not found

def compile_data():

    main_df = pd.DataFrame()
    
    for count, ticker in enumerate(tickers):
        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
        df.set_index('Date',inplace=True)
        
        df.rename(columns = {'Adj Close': ticker}, inplace=True)
        df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)
        
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')
            
        if count % 10 == 0:
            print(count)
            
            
    print(main_df.head())
    main_df.to_csv('joined_data_other.csv')
    
compile_data()



df = pd.read_csv('joined_data_other.csv')

df = df[df['Date'] > '2015-01-01']

df.dropna(axis=1, how="any", inplace=True) #, thresh=None, subset=None, inplace=False)


stock_names = list(df.columns)
stock_names.pop(stock_names.index('Date'))

tickers = stock_names

#df['AAPL'].plot()
#plt.show()
df_corr = df.corr() # VERY WELL PAID INFO! informs stock picking
# could do this with a delay as well and get more results



print(df_corr.head())


# save stock tickers and turn df into numpy array
#stock_names = df_corr.columns.tolist()

corr = df_corr.to_numpy()

# turn to array, save company names in order and find the locations
# of the values that are close to zero, etc and return the pairs
# turn to sets if I want groups
# np.where(a == 3)[0]

# plot a distribution of values so I can see which cutoffs make sense
# organize into pairs and 

n = corr.shape[0]
corr[range(n), range(n)] = np.nan

# no need for pairs, just remove 1 values from diagonal
# and then flatten and plot

corr_flat = np.reshape(corr, [1,n*n])
# maybe remove nans
corr_flat = corr_flat[~np.isnan(corr_flat)]


#plt.hist(corr_flat, bins = [ -1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1 ])
#plt.show()
#plt.savefig('histogram_correlations.png', dpi=1000)


num_bins = 10

fig, ax = plt.subplots()

# the histogram of the data
n, bins, patches = ax.hist(corr_flat, num_bins, density=True, width=0.15)

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.show()
plt.savefig('histogram_correlations.png', dpi=1000)





uncorr_inds = np.where((corr > -0.2) & (corr < 0.2))
pos_cor_inds = np.where(corr > 0.95) 
# 0.75-0.85 makes all of them correlated, at least the sp500 ones. make it stricter
# 0.95 shows some groups. but try different than sp500, maybe these are all highly correlated
# or play around more with other stuff like correlating fewer years
neg_cor_inds = np.where(corr < -0.75)


uncorr_x = list(uncorr_inds[0])
uncorr_y = list(uncorr_inds[1])
uncorr_pairs = [[stock_names[x],stock_names[y]] for (x,y) in zip(uncorr_x,uncorr_y) ]
print('uncor_pairs len: ', len(uncorr_pairs))

pos_cor_x = list(pos_cor_inds[0])
pos_cor_y = list(pos_cor_inds[1])
pos_cor_pairs = [[stock_names[x],stock_names[y]] for (x,y) in zip(pos_cor_x,pos_cor_y) ]
print('pos cor pairs len: ', len(pos_cor_pairs))    

neg_cor_x = list(neg_cor_inds[0])
neg_cor_y = list(neg_cor_inds[1])
neg_cor_pairs = [[stock_names[x],stock_names[y]] for (x,y) in zip(neg_cor_x,neg_cor_y) ]
print('neg cor pairs len: ', len(neg_cor_pairs))   

# how to reduce these lists to groups? transitive and symmetric property apply
# removing symmetric pairs is easy


import networkx

samples = pos_cor_pairs

samples = [pair for pair in pos_cor_pairs if pair[0] != pair[1]]
# eliminate the pairs of stocks correlated with themselves


ga = networkx.Graph(samples)
conn_comp = networkx.connected_components(ga)
groups = []

for subgraph in networkx.connected_components(ga):
    groups.append(subgraph)


# now turn this into a dataframe with an index for groups, and then find their sectors

# turn list of groups into list of lists
groups_list = [list(group) for group in groups]

grouped_stocks = []

for i in range(0,len(groups_list)):

    cur_group = groups_list[i]    

    if len(cur_group) == 1:
        grouped_stocks.append([cur_group[0], i])
        
    else:
        whole_group = [[tick, i] for tick in cur_group]
        grouped_stocks = grouped_stocks + whole_group


#import itertools
#grouped_stocks_unpacked = list(itertools.chain.from_iterable(grouped_stocks))


# save this offline, takes a long time to find subgroups

#tickers = [item[0] for item in grouped_stocks]  
#tickers2 = Ticker(tickers)

data_other = pd.DataFrame(grouped_stocks, columns=['Ticker', 'Corr group']) 

data_other.to_csv('other_stocks_grouped.csv')



data_other = pd.read_csv('other_stocks_grouped.csv')

data_other.drop(columns=['Unnamed: 0'], inplace=True)


#tickers = list(data_other['Ticker'].unique())

#tickers2 = Ticker(tickers)



dictionary = {'Ticker':[],
              'Country':[],
              'Industry':[],
              'Sector':[],
              'Industry key':[],
              'Sector key':[]}



# sometimes API might not connect properly

for tick in tickers:
    
    print('now')
    dictionary['Ticker'].append(tick)
    
    t = Ticker(tick)
    
    try:
        asset_info = t.asset_profile
        info = asset_info[tick]
        
        try:
            dictionary['Country'].append(info['country'])
        except:
            dictionary['Country'].append(np.nan)
            
        
        try:
            dictionary['Industry'].append(info['industry'])
        except:
            dictionary['Industry'].append(np.nan)
                
        try:
            dictionary['Sector'].append(info['sector'])
        except:
            dictionary['Sector'].append(np.nan)
                    
        try:
            dictionary['Industry key'].append(info['industryKey'])
        except:
            dictionary['Industry key'].append(np.nan)
                        
        try:
            dictionary['Sector key'].append(info['sectorKey'])
        except:
            dictionary['Sector key'].append(np.nan)

    except:
        print('error with ticker: ', tick) 
        
        dictionary['Country'].append(np.nan)
        dictionary['Industry'].append(np.nan)
        dictionary['Sector'].append(np.nan)
        dictionary['Industry key'].append(np.nan)
        dictionary['Sector key'].append(np.nan)
        

df_info = pd.DataFrame.from_dict(dictionary)

sector_data = data_other.merge(df_info, how='outer', on=['Ticker'])

company_data = other.merge(sector_data, how='right', on=['Ticker'])

# merge company data, group data and company name
# on the subset of tickers we selected


company_data.to_csv('corr_company_data.csv')



company_data = pd.read_csv('corr_company_data.csv')

group_sizes = company_data['Corr group'].value_counts()

# group by group and see if they tend to belong in the same country or sector

countries = company_data[['Security Name', 'Corr group', 'Country']]

sectors = company_data[['Security Name', 'Corr group', 'Sector']]


def grouped_variables(data,variable):

    data = data.set_index('Security Name')
    data = data.groupby('Corr group')[variable].apply(list)
    
    variable_lists = list(data)
    
    #new_variable_lists = [[x for x in item if str(x) != 'nan'] for item in variable_lists]
    
    unique_list = [pd.Series(item).drop_duplicates().tolist() for item in variable_lists]
    
    return unique_list
    

grouped_countries = grouped_variables(countries, 'Country')
grouped_sectors = grouped_variables(sectors, 'Sector')

groups_df = {'# tickers' : list(group_sizes), 
             'countries': grouped_countries, 
             'sectors': grouped_sectors}

groups_df = pd.DataFrame.from_dict(groups_df)

groups_df.to_csv('grouped_countries_sectors.csv')



