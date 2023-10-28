#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 17:00:00 2023

@author: Dimitra Bourou
"""

import yahoo_fin.stock_info as si
# the package is not very well maintained, last update 2021 and now tons of errors
# later susbstitute with another one, there are alternatives
from yahooquery import Ticker

# set current folder to "outputs" so that CSVs we generate get stored there

from pathlib import Path
import os

root_dir = Path(__file__).resolve().parent
path_out = os.path.join(root_dir, 'outputs')
os.chdir(path_out)
cd = os.getcwd()


# get some common lists of stocks
# uncomment to select another, now using dow
stocks_list = si.tickers_dow()
#stocks_list = si.tickers_nasdaq()
#stocks_list = si.tickers_other()
#stocks_list = si.tickers_sp500()

# get all the valuation information about the stocks in the list

tickers = Ticker(stocks_list)
combined_stats = tickers.valuation_measures

# columns appearing in combined stats:
# ['asOfDate', 'periodType', 'EnterpriseValue',
#       'EnterprisesValueEBITDARatio', 'EnterprisesValueRevenueRatio',
#       'ForwardPeRatio', 'MarketCap', 'PbRatio', 'PeRatio', 'PegRatio',
#       'PsRatio']


# stats for the same ticker are included for several dates and also calculated for 3M or TTM, choose TTM
combined_stats = combined_stats[combined_stats['periodType'] == 'TTM']

# remove columns we aren't interested in
combined_stats.drop([ 'periodType', 'EnterpriseValue','ForwardPeRatio','MarketCap' ], 1, inplace=True)

# apply the heuristic criteria value investors use to determine if a stock has "a good value"
bool1 = combined_stats['PeRatio'] < 25.0
bool2 = combined_stats['PegRatio'] < 1.0
bool3 = combined_stats['PsRatio'] < 2.0
bool4 = combined_stats['PbRatio'] < 1.0
bool5 = combined_stats['EnterprisesValueRevenueRatio'] < 3.0
bool6 = combined_stats['EnterprisesValueEBITDARatio'] < 10.0

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

# sum the boolean columns to get a "valuation index" from 0 to 6
# the more criteria a stock satisfies, the more of a good bargain it is
combined_stats['val score'] = combined_stats['good pe'] + combined_stats['good peg'] + combined_stats['good ps'] + combined_stats['good pb'] + combined_stats['good evr'] + combined_stats['good eve']

# create a new dataframe called "valuations" with only tickers and valuation scores
# get ticker as column and not as index, then sort by valuation scores
valuations = combined_stats
valuations.reset_index(inplace=True)
valuations = valuations[['symbol','asOfDate','val score']]
valuations = valuations.sort_values('val score',ascending=False)
valuations.reset_index(inplace=True)
valuations.drop(columns=['index'],inplace=True)

# save dataframe as csv
valuations.to_csv('stock_valuations_rank.csv')

# create a separate dataframe for PE ratios specifically (most common valuation metric)
# and sort stocks by it, from low (good value) to high
pe_ratios = combined_stats['PeRatio'].reset_index()
pe_ratios = pe_ratios.sort_values('PeRatio',ascending=True)
pe_ratios.reset_index(inplace=True)
pe_ratios.drop(columns=['index','level_0'],inplace=True)

# save PE ratios to csv
pe_ratios.to_csv('stocks_pe_ratios_rank.csv')




