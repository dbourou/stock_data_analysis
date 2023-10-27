#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 08:58:56 2023

@author: Dimitra Bourou
"""

import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('QtAgg')
from matplotlib import style

import yahoo_fin.stock_info as si
import yahoo_fin.news as news
# the package is not very well maintained, last update 2021 and now tons of errors
# later susbstitute with another one, there are alternatives

from yahooquery import Ticker


style.use('ggplot')



# get some common lists of stocks
dow = si.tickers_dow()
nasdaq = si.tickers_nasdaq()
other = si.tickers_other()
sp500 = si.tickers_sp500()

mystocks = ['biib', 'regn', 'aixa', 'una', 'meta', 'baba', 'vnm', 'ech' ]

tobuy = ['ewm','thd', 'givn' ]
# malaysian etf, thai etf, givadaun

# can I get etf info?


quote_table = si.get_quote_table("aapl", dict_result=True)
#returns dictionary
quote_table["PE Ratio (TTM)"]
            
quote_table = si.get_quote_table("aapl", dict_result=False)
# returns dataframe

#quote_table["PE Ratio (TTM)"]


si.get_stats_valuation("aapl")


# get list of Dow tickers
dow_list = si.tickers_dow()

dow_stats = {}
for ticker in dow_list:
    temp = si.get_stats_valuation(ticker)
    temp = temp.iloc[:,:2]
    temp.columns = ["Attribute", "Recent"]
    dow_stats[ticker] = temp

combined_stats = pd.concat(dow_stats)
combined_stats = combined_stats.reset_index()

#combined_stats['Attribute'].unique() = 
#['Market Cap (intraday)', 'Enterprise Value', 
#'Trailing P/E','Forward P/E', 'PEG Ratio (5 yr expected)', 'Price/Sales (ttm)',
#'Price/Book (mrq)', 'Enterprise Value/Revenue', 'Enterprise Value/EBITDA']

# somehow compare similar variables and get "cheapest" companies
#combined_stats.sort_values(['Attribute', 'Recent'], ascending=[True, True])

# remove market cap and enterprise value and turn the rest to floats
combined_stats = combined_stats[~ (combined_stats['Attribute'] == 'Market Cap (intraday)')] 
combined_stats = combined_stats[~ (combined_stats['Attribute'] == 'Enterprise Value')]

#combined_stats['Recent'].astype('float',copy=False)
# there are nans, float didnt work
combined_stats['Recent']=combined_stats['Recent'].astype("float64", errors="ignore")
#combined_stats['Recent']=combined_stats['Recent'].apply(pd.to_numeric, errors='coerce')

bool1 = (combined_stats['Attribute'] == 'Trailing P/E') & (combined_stats['Recent'] < 25.0)
bool2 = (combined_stats['Attribute'] == 'PEG Ratio (5 yr expected)') & (combined_stats['Recent'] < 1.0)
bool3 = (combined_stats['Attribute'] == 'Price/Sales (ttm)') & (combined_stats['Recent'] < 2.0)
bool4 = (combined_stats['Attribute'] == 'Price/Book (mrq)' ) & (combined_stats['Recent'] < 1.0)
bool5 = (combined_stats['Attribute'] == 'Enterprise Value/Revenue') & (combined_stats['Recent'] < 3.0)
bool6 = (combined_stats['Attribute'] == 'Enterprise Value/EBITDA' ) & (combined_stats['Recent'] < 10.0)

# add columns named "good pe", "good pb" with number 1 or 0 and then group by ticker 
# and sum to get a "valuation" score
combined_stats['good pe'] = 0.0
combined_stats['good peg'] = 0.0
combined_stats['good ps'] = 0.0
combined_stats['good pb'] = 0.0
combined_stats['good evr'] = 0.0
combined_stats['good eve'] = 0.0

combined_stats.loc[bool1, 'good pe'] = 1.0
combined_stats.loc[bool2, 'good peg'] = 1.0
combined_stats.loc[bool3, 'good ps'] = 1.0
combined_stats.loc[bool4, 'good pb'] = 1.0
combined_stats.loc[bool5, 'good evr'] = 1.0
combined_stats.loc[bool6, 'good eve'] = 1.0

del combined_stats["level_1"]
# update column names
#combined_stats.columns = ["Ticker", "Attribute", "Recent"]

combined_stats['val score'] = combined_stats['good pe'] + combined_stats['good peg'] + combined_stats['good ps'] + combined_stats['good pb'] + combined_stats['good evr'] + combined_stats['good eve']
# DOESNT WORK, WONT BE ALL IN SAME COLUMNS. MUST GROUP BY TICKER AND AGGREGATE
# or fill nans with zeros in the "good" columns, also works
# must still group and aggregate
valuations = combined_stats[['level_0', 'val score']]
valuations = valuations.groupby('level_0').agg({'val score': 'sum'})
valuations.reset_index(inplace=True)
valuations.rename(columns={'level_0':'Ticker'},inplace=True)
valuations = valuations.sort_values('val score',ascending=False)
valuations.reset_index(inplace=True)
valuations.drop(columns=['index'],inplace=True)

# cleanup combined stats
combined_stats.rename(columns={'level_0':'Ticker','Recent':'Value'},inplace=True)
combined_stats = combined_stats[['Ticker','Attribute','Value']]

pe_ratios = combined_stats[combined_stats["Attribute"]=="Trailing P/E"].reset_index()
pe_ratios = pe_ratios.sort_values('Value',ascending=True)
pe_ratios.reset_index(inplace=True)
pe_ratios.drop(columns=['index'],inplace=True)







