#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 17:00:00 2023

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
# uncomment to select another, now using dow
stocks_list = si.tickers_dow()
#stocks_list = si.tickers_nasdaq()
#stocks_list = si.tickers_other()
#stocks_list = si.tickers_sp500()

# get all the valuation information about the stocks in the list

stock_stats = {}
for ticker in stocks_list:
    temp = si.get_stats_valuation(ticker)
    temp = temp.iloc[:,:2]
    temp.columns = ["Attribute", "Recent"]
    stock_stats[ticker] = temp

combined_stats = pd.concat(stock_stats)
combined_stats = combined_stats.reset_index()

# different attributes appearing in combined stats:
#'Market Cap (intraday)', 'Enterprise Value', 
#'Trailing P/E','Forward P/E', 'PEG Ratio (5 yr expected)', 'Price/Sales (ttm)',
#'Price/Book (mrq)', 'Enterprise Value/Revenue', 'Enterprise Value/EBITDA'

# remove market cap and enterprise value and turn the rest to floats
combined_stats = combined_stats[~ (combined_stats['Attribute'] == 'Market Cap (intraday)')] 
combined_stats = combined_stats[~ (combined_stats['Attribute'] == 'Enterprise Value')]
combined_stats['Recent']=combined_stats['Recent'].astype("float64", errors="ignore")

# apply the heuristic criteria value investors use to determine if a stock has "a good value"
bool1 = (combined_stats['Attribute'] == 'Trailing P/E') & (combined_stats['Recent'] < 25.0)
bool2 = (combined_stats['Attribute'] == 'PEG Ratio (5 yr expected)') & (combined_stats['Recent'] < 1.0)
bool3 = (combined_stats['Attribute'] == 'Price/Sales (ttm)') & (combined_stats['Recent'] < 2.0)
bool4 = (combined_stats['Attribute'] == 'Price/Book (mrq)' ) & (combined_stats['Recent'] < 1.0)
bool5 = (combined_stats['Attribute'] == 'Enterprise Value/Revenue') & (combined_stats['Recent'] < 3.0)
bool6 = (combined_stats['Attribute'] == 'Enterprise Value/EBITDA' ) & (combined_stats['Recent'] < 10.0)

# add boolean columns denoting how many of the above criteria a ticker satisfies
# first fill them with zeros
combined_stats['good pe'] = 0.0
combined_stats['good peg'] = 0.0
combined_stats['good ps'] = 0.0
combined_stats['good pb'] = 0.0
combined_stats['good evr'] = 0.0
combined_stats['good eve'] = 0.0

# now add "1" only to tickers where the respective criteria are satisfied
combined_stats.loc[bool1, 'good pe'] = 1.0
combined_stats.loc[bool2, 'good peg'] = 1.0
combined_stats.loc[bool3, 'good ps'] = 1.0
combined_stats.loc[bool4, 'good pb'] = 1.0
combined_stats.loc[bool5, 'good evr'] = 1.0
combined_stats.loc[bool6, 'good eve'] = 1.0

del combined_stats["level_1"]

# sum the boolean columns to get a "valuation index" from 0 to 6
# the more criteria a stock satisfies, the more of a good bargain it is
combined_stats['val score'] = combined_stats['good pe'] + combined_stats['good peg'] + combined_stats['good ps'] + combined_stats['good pb'] + combined_stats['good evr'] + combined_stats['good eve']

valuations = combined_stats[['level_0', 'val score']]
valuations = valuations.groupby('level_0').agg({'val score': 'sum'})
valuations.reset_index(inplace=True)
valuations.rename(columns={'level_0':'Ticker'},inplace=True)
valuations = valuations.sort_values('val score',ascending=False)
valuations.reset_index(inplace=True)
valuations.drop(columns=['index'],inplace=True)

# cleanup combined stats and rename columns
combined_stats.rename(columns={'level_0':'Ticker','Recent':'Value'},inplace=True)
combined_stats = combined_stats[['Ticker','Attribute','Value']]

# create a separate dataframe for PE ratios specifically (most common valuation metric)
# and sort stocks by it, from low to high
pe_ratios = combined_stats[combined_stats["Attribute"]=="Trailing P/E"].reset_index()
pe_ratios = pe_ratios.sort_values('Value',ascending=True)
pe_ratios.reset_index(inplace=True)
pe_ratios.drop(columns=['index'],inplace=True)







