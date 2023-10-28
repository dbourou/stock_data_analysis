This repository contains scripts for importing
historical stock price data, as well as fundamentals
or valuation metrics about companies from yahoo finance (using APIs).

We can then use this data to do any analyses we want.

In the two scripts present here I do the following:

1. find_value_stocks

Here I get the valuation metrics for hundreds of companies (stocks)
from yahoo finance and filter them using certain heuristics
value investors use to determine if a company is a good "bargain".
I assign a "value score" from 1 to 6 based on how many of these
criteria the company fullfills.

2. correlated_stock_groups_sectors

We get historical price data from all companies found in
"other" stock exchanges, according to yahoo finance,
calculate a covariance matrix based on several years of data,
and then use a Python package called networkX
to turn the pairs of highly correlated stocks into
groups of highly correlated stocks

( To do: get data from sectors and countries of companies and
see if correlated groups tend to be from the same ones
Also, investigate why the majority of stocks tend to be very highly correlated; is this expected? )

