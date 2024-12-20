import pandas as pd
import numpy as np
from pandas_ta import rsi, bbands, atr, macd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import yfinance as yf
import pandas_datareader.data as web

# 1. Download stock data
def load_stock_data(tickers, start_date, end_date):
    df = yf.download(tickers=tickers, start=start_date, end=end_date).stack()
    df.index.names = ['date', 'ticker']
    df.columns = df.columns.str.lower()
    return df

# 2. Calculate technical indicators
def calculate_indicators(df):
    df['garman_klass_vol'] = ((np.log(df['high'])-np.log(df['low']))**2)/2-(2*np.log(2)-1)*((np.log(df['adj close'])-np.log(df['open']))**2)

    df['rsi'] = df.groupby(level=1)['adj close'].transform(lambda x: rsi(close=x, length=20))

    df['bb_low'] = df.groupby(level=1)['adj close'].transform(lambda x: bbands(close=np.log1p(x), length=20).iloc[:,0])
                                                            
    df['bb_mid'] = df.groupby(level=1)['adj close'].transform(lambda x: bbands(close=np.log1p(x), length=20).iloc[:,1])
                                                            
    df['bb_high'] = df.groupby(level=1)['adj close'].transform(lambda x: bbands(close=np.log1p(x), length=20).iloc[:,2])

    def compute_atr(stock_data):
        atr = atr(high=stock_data['high'],
                            low=stock_data['low'],
                            close=stock_data['close'],
                            length=14)
        return atr.sub(atr.mean()).div(atr.std())

    df['atr'] = df.groupby(level=1, group_keys=False).apply(compute_atr)

    def compute_macd(close):
        macd = macd(close=close, length=20).iloc[:,0]
        return macd.sub(macd.mean()).div(macd.std())

    df['macd'] = df.groupby(level=1, group_keys=False)['adj close'].apply(compute_macd)

    df['dollar_volume'] = (df['adj close']*df['volume'])/1e6
    
    return df

# 3. Aggregate to monthly data
def aggregate_to_monthly(df, number_of_stocks):
    last_cols = [c for c in df.columns if c not in ['dollar_volume', 'volume', 'open', 'high', 'low', 'close']]
    data = pd.concat([
        df.unstack('ticker')['dollar_volume'].resample('M').mean().stack('ticker').to_frame('dollar_volume'),
        df[last_cols].unstack('ticker').resample('M').last().stack('ticker')
    ], axis=1).dropna()
    
    data['dollar_volume'] = (data.loc[:, 'dollar_volume'].unstack('ticker').rolling(5*12, min_periods=12).mean().stack())

    data['dollar_vol_rank'] = (data.groupby('date')['dollar_volume'].rank(ascending=False))

    number_to_drop = int(number_of_stocks * 0.3)

    data = data[data['dollar_vol_rank']<number_to_drop].drop(['dollar_volume', 'dollar_vol_rank'], axis=1)

    return data

# 4. Calculate Monthly Returns for different time horizons as features
def calculate_returns(df):

    outlier_cutoff = 0.005

    lags = [1, 2, 3, 6, 9, 12]

    for lag in lags:

        df[f'return_{lag}m'] = (df['adj close']
                              .pct_change(lag)
                              .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                                     upper=x.quantile(1-outlier_cutoff)))
                              .add(1)
                              .pow(1/lag)
                              .sub(1))
    return df
    
    
# 5. Calculate rolling factor betas
def calculate_factor_betas(data, start_date):
    factor_data = web.DataReader('F-F_Research_Data_5_Factors_2x3',
                               'famafrench',
                               start=start_date)[0].drop('RF', axis=1)
    factor_data.index = factor_data.index.to_timestamp()

    factor_data = factor_data.resample('M').last().div(100)

    factor_data.index.name = 'date'

    factor_data = factor_data.join(data['return_1m']).sort_index()

    # - Filter out stocks with less than 10 months of data

    observations = factor_data.groupby(level=1).size()

    valid_stocks = observations[observations >= 10]

    factor_data = factor_data[factor_data.index.get_level_values('ticker').isin(valid_stocks.index)]

    # - Calculate Rolling Factors Betas

    betas = (factor_data.groupby(level=1,
                                group_keys=False)
            .apply(lambda x: RollingOLS(endog=x['return_1m'], 
                                        exog=sm.add_constant(x.drop('return_1m', axis=1)),
                                        window=min(24, x.shape[0]),
                                        min_nobs=len(x.columns)+1)
            .fit(params_only=True)
            .params
            .drop('const', axis=1)))

    return betas


def load_and_create_all_features(tickers, start_date, end_date, number_of_stocks):
    df = load_stock_data(tickers, start_date, end_date)
    df = calculate_indicators(df)
    df = aggregate_to_monthly(df, number_of_stocks)
    df = df.groupby(level=1, group_keys=False).apply(calculate_returns).dropna()

    betas = calculate_factor_betas(df, start_date)
    factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']

    df = (df.join(betas.groupby('ticker').shift()))

    df.loc[:, factors] = df.groupby('ticker', group_keys=False)[factors].apply(lambda x: x.fillna(x.mean()))

    df = df.drop('adj close', axis=1)

    df = df.dropna()

    return df

