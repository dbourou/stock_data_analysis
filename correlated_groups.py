#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 17:00:00 2023

@author: Dimitra Bourou
"""

import datetime as dt
import os
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')


import matplotlib
matplotlib.use('QtAgg')
from matplotlib import style

import yahoo_fin.stock_info as si
# the package is not very well maintained, last update 2021 and now tons of errors
# later susbstitute with another one, there are alternatives
from yahooquery import Ticker


style.use('ggplot')


# get some common lists of stocks
other = si.tickers_other(include_company_data=True) # but idk which stockexchange they are from, get company info
# other = si.tickers_dow()
# other = si.tickers_nasdaq()
# other = si.tickers_sp500()
# to get tickers from another stock exchange, uncomment one of the previous 3 lines
# the script might then require small modifications

# there are many different versions of the tickers
# for simplicity just keep companies that have the same ticker in all 3, they are the majority

a1 = ( other['ACT Symbol'] == other['NASDAQ Symbol'] )
a2 = ( other['ACT Symbol'] == other['CQS Symbol'] )
a3 = ( other['CQS Symbol'] == other['NASDAQ Symbol'] )

other = other.loc[(a1 & a2 & a3)] 
other = other[['ACT Symbol', 'Security Name']]
other.rename(columns={'ACT Symbol': 'Ticker'}, inplace=True)

tickers = list(other['Ticker'].unique())

##### REMOVE THIS AFTER
tickers = tickers[0:200]

# get the historical stock price data, and save them locally (this will take some time)
# make also an "outputs folder" to store this and other data

# first check if the directory exists to save the outputs
isExist = os.path.exists('outputs')
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs('outputs')
   
# first check if the directory exists to save the stock data
path = os.path.join('outputs', 'stock_dfs')
isExist = os.path.exists(path)
if not isExist:
      # Create a new directory because it does not exist
      os.makedirs(path)
      
      
# set current folder to "outputs" so that CSVs we generate get stored there

from pathlib import Path

root_dir = Path(__file__).resolve().parent
path_out = os.path.join(root_dir, 'outputs')
os.chdir(path_out)
cd = os.getcwd()

# download the stock data locally

start = dt.datetime(2015,1,1)
end = dt.datetime(2023,1,1)

for ticker in tickers: # take all tickers later, will take a few mins
    print(ticker)    
    if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
        df = yf.download(ticker, start=start, end=end)
        df.to_csv('stock_dfs/{}.csv'.format(ticker))
    else:
        print('Already have {}'.format(ticker))


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

df.dropna(axis=1, how="any", inplace=True)


stock_names = list(df.columns)
stock_names.pop(stock_names.index('Date'))

tickers = stock_names

# get the covariance matrix
df_corr = df.corr()

# turn df into numpy array

corr = df_corr.to_numpy()

# plot a distribution of correlation values 
# to see how the stocks are distributed in terms of how correlated they are
# and to see which cutoffs make sense (i.e., if almost all have 0.8 correlation
# but very few have 0.85 then thats a good threshold)
# unfortunately in this case almost all stocks have correlation close to 1

# first replace the diagonal values with nans, we don't want those
n = corr.shape[0]
corr[range(n), range(n)] = np.nan

# flatten the covariance matrix (we just want the values) and make a barplot
corr_flat = np.reshape(corr, [1,n*n])
# maybe remove nans
corr_flat = corr_flat[~np.isnan(corr_flat)]

num_bins = 10

fig, ax = plt.subplots()

# the histogram of the data
n, bins, patches = ax.hist(corr_flat, num_bins, density=True, width=0.15)

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.show()
plt.savefig('histogram_correlations.png', dpi=1000)

# find the indices in the covariance matrix where cov > 0.95
pos_cor_inds = np.where(corr > 0.95) 

# save the tickers of the correlated stocks as a list of pairs
pos_cor_x = list(pos_cor_inds[0])
pos_cor_y = list(pos_cor_inds[1])
pos_cor_pairs = [[stock_names[x],stock_names[y]] for (x,y) in zip(pos_cor_x,pos_cor_y) ]
print('pos cor pairs len: ', len(pos_cor_pairs))    

# however, positive correlation is a symmetric and transitive property,
# meaning that if A corr with B and B with C we have a group ABC of corr stocks
# to get these groups we can use network theory
# each ticker pair is a node in a graph and we can find connected components

import networkx

samples = pos_cor_pairs

samples = [pair for pair in pos_cor_pairs if pair[0] != pair[1]]
# exclude the pairs of stocks correlated with themselves

ga = networkx.Graph(samples)
conn_comp = networkx.connected_components(ga)
groups = []

for subgraph in networkx.connected_components(ga):
    groups.append(subgraph)


# now turn this into a dataframe with an index for groups

groups_list = [list(group) for group in groups]

grouped_stocks = []

for i in range(0,len(groups_list)):

    cur_group = groups_list[i]    

    if len(cur_group) == 1:
        grouped_stocks.append([cur_group[0], i])
        
    else:
        whole_group = [[tick, i] for tick in cur_group]
        grouped_stocks = grouped_stocks + whole_group


# save tickers and their groups into a CSV, because the network algorithm can take a long time to run
# so if we want to add subsequent analyses from here onward, good to reload the CSV we saved
data_other = pd.DataFrame(grouped_stocks, columns=['Ticker', 'Corr group']) 

data_other.to_csv('other_stocks_grouped.csv')

data_other = pd.read_csv('other_stocks_grouped.csv')

# create a dataframe where each row is one group of correlated stocks (index is group number),
# and all correlated stocks are a list
data_other = data_other.groupby('Corr group')['Ticker'].apply(list)






