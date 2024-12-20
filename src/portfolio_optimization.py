from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import pandas as pd
import yfinance as yf
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import streamlit as st


def optimize_weights(prices, lower_bound=0):
    
    returns = expected_returns.mean_historical_return(prices=prices,
                                                      frequency=252)
    
    cov = risk_models.sample_cov(prices=prices,
                                 frequency=252)
    
    ef = EfficientFrontier(expected_returns=returns,
                           cov_matrix=cov,
                           weight_bounds=(lower_bound, .1),
                           solver='SCS')
    
    weights = ef.max_sharpe()
    
    return ef.clean_weights()

def tickers_for_each_month(df):
    filtered_df = df[df['cluster']==3].copy()

    filtered_df = filtered_df.reset_index(level=1)

    filtered_df.index = filtered_df.index+pd.DateOffset(1)

    filtered_df = filtered_df.reset_index().set_index(['date', 'ticker'])

    dates = filtered_df.index.get_level_values('date').unique().tolist()

    fixed_dates = {}

    for d in dates:
        
        fixed_dates[d.strftime('%Y-%m-%d')] = filtered_df.xs(d, level=0).index.tolist()
        
    del fixed_dates['2025-01-01']

    return fixed_dates

def download_portfolio_ticker_daily_prices(df):
    stocks = df.index.get_level_values('ticker').unique().tolist()

    new_df = yf.download(tickers=stocks,
                        start=df.index.get_level_values('date').unique()[0]-pd.DateOffset(months=12),
                        end=df.index.get_level_values('date').unique()[-1])

    return new_df

def get_portfolio_returns(df, fixed_dates):

    returns_dataframe = np.log(df['Adj Close']).diff()

    portfolio_df = pd.DataFrame()

    for start_date in fixed_dates.keys():
        try:

            end_date = (pd.to_datetime(start_date)+pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')

            cols = fixed_dates[start_date]

            optimization_start_date = (pd.to_datetime(start_date)-pd.DateOffset(months=12)).strftime('%Y-%m-%d')

            optimization_end_date = (pd.to_datetime(start_date)-pd.DateOffset(days=1)).strftime('%Y-%m-%d')
            
            optimization_df = df['Adj Close'][cols][optimization_start_date:optimization_end_date]

            success = False
            try:
                weights = optimize_weights(prices=optimization_df,
                                    lower_bound=round(1/(len(optimization_df.columns)*2),3))

                weights = pd.DataFrame(weights, index=pd.Series(0))
                
                success = True
            except:
                print(f'Max Sharpe Optimization failed for {start_date}, Continuing with Equal-Weights')
            
            if success==False:
                weights = pd.DataFrame([1/len(optimization_df.columns) for i in range(len(optimization_df.columns))],
                                        index=optimization_df.columns.tolist(),
                                        columns=pd.Series(0)).T
            
            temp_df = returns_dataframe[start_date:end_date]
        
            temp_df = temp_df.stack().to_frame('return').reset_index(level=0)\
                    .merge(weights.stack().to_frame('weight').reset_index(level=0, drop=True),
                            left_index=True,
                            right_index=True)\
                    .reset_index().set_index(['Date', 'Ticker']).unstack().stack()

            temp_df['weighted_return'] = temp_df['return']*temp_df['weight']

            temp_df = temp_df.groupby(level=0)['weighted_return'].sum().to_frame('Strategy Return')
            
            portfolio_df = pd.concat([portfolio_df, temp_df], axis=0)
            
        except Exception as e:
            print(e)
            
    portfolio_df = portfolio_df.drop_duplicates()
    return portfolio_df


def common_index_returns(portfolio_df, start_date, index_choice):
    if index_choice == 'NASDAQ 100':
        qqq = yf.download(tickers='QQQ',
                    start=start_date,
                    end=dt.date.today())

        qqq_ret = np.log(qqq[['Adj Close']]).diff().dropna().rename({'Adj Close':'QQQ Buy&Hold'}, axis=1)
        qqq_ret.columns = qqq_ret.columns.get_level_values(0)
        portfolio_df = portfolio_df.merge(qqq_ret,
                                        left_index=True,
                                        right_index=True)   
    
    elif index_choice == 'Dow Jones':
        dia = yf.download(tickers='DIA',
                    start=start_date,
                    end=dt.date.today())

        dia_ret = np.log(dia[['Adj Close']]).diff().dropna().rename({'Adj Close':'DOW Jones Buy&Hold'}, axis=1)
        dia_ret.columns = dia_ret.columns.get_level_values(0)
        portfolio_df = portfolio_df.merge(dia_ret,
                                        left_index=True,
                                        right_index=True)   

    else:    
        spy = yf.download(tickers='SPY',
                    start=start_date,
                    end=dt.date.today())


        spy_ret = np.log(spy[['Adj Close']]).diff().dropna().rename({'Adj Close':'SPY Buy&Hold'}, axis=1)
        spy_ret.columns = spy_ret.columns.get_level_values(0)

        portfolio_df = portfolio_df.merge(spy_ret,
                                        left_index=True,
                                        right_index=True)

    return portfolio_df

def draw_graph(portfolio_df):
    plt.style.use('ggplot')
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    portfolio_cumulative_return = np.exp(np.log1p(portfolio_df).cumsum()) - 1
    
    portfolio_cumulative_return[:dt.date.today()].plot(ax=ax)
    
    ax.set_title('Unsupervised Learning Trading Strategy Returns Over Time')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax.set_ylabel('Return')
    
    return fig


def run_portfolio_optimization(df, displayed_start_date, index_choice):
    fixed_dates = tickers_for_each_month(df)
    daily_tickers_df = download_portfolio_ticker_daily_prices(df)
    portfolio_df = get_portfolio_returns(daily_tickers_df, fixed_dates)
    all_returns_df = common_index_returns(portfolio_df, displayed_start_date, index_choice)
    return all_returns_df
