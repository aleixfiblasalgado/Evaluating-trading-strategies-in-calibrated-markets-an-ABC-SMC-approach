"""
## Simulator
@author: aleix
"""

import numpy as np
import pandas as pd
import datetime
import os
import itertools
from util.LOBSTER_utils import get_trades, stream_cleaning
from util.LOBSTER_utils import get_interarrival_times, get_orderbook_stylised_facts

from metrics.minutely_returns import MinutelyReturns
from metrics.autocorrelation_m import Autocorrelation
from metrics.volatility_clustering_m import VolatilityClustering

datapath = "../data/_data_dwn_16_372__AAPL_2021-11-22_2021-11-29_20"
ob_path = os.path.join(datapath, "AAPL_2021-11-22_24900000_57900000_orderbook_20.csv")
stream_path = os.path.join(datapath, "AAPL_2021-11-22_24900000_57900000_message_20.csv")
    
def get_market_data(stream_path, ob_path, start_time , end_time):
    orderbook = pd.read_csv(ob_path, header=None, index_col=None)
    stream = pd.read_csv(stream_path, header=None, index_col=None)
    
    ## stream cleansing
    stream.drop(6, inplace=True, axis = 1)
    stream.columns=["Time (sec)", "Event Type", "Order ID", "Size", "Price", "Direction"]
    stream['QuoteTime'] = datetime.datetime.fromisoformat('2021-11-22T00:00:00') + pd.to_timedelta(stream.iloc[:,0], 's')
    stream = stream.drop('Time (sec)', axis=1)
    first_column = stream.pop('QuoteTime')
    stream.insert(0, 'QuoteTime', first_column)
    
    ## ob cleansing
    num_levels = 20
    columns = list(itertools.chain(
        *[[f'ask_price_{level}', f'ask_size_{level}', f'bid_price_{level}', f'bid_size_{level}'] for level in 
          range(1, num_levels + 1)]))
    orderbook.columns = columns
    
    ohlcv = get_trades(stream, start_time, end_time)
    
    ## Asset returns stylized facts
    all_metrics = [MinutelyReturns, Autocorrelation, VolatilityClustering]
    metric_names = ["MinutelyReturns", "Autocorrelation", "VolatilityClustering"]    
    result = dict()
    
    for my_metric, metric_name in zip(all_metrics, metric_names):
        my_metric = my_metric()
        result[metric_name] = np.array(my_metric.compute(ohlcv))

    result.update({'VolatilityClustering':result['VolatilityClustering'].flatten()})
    
    stream_processed = stream_cleaning(stream, start_time, end_time)
    ob_levels=3
    
    ## Volume stylized facts
    obs = get_orderbook_stylised_facts(stream_processed, orderbook, num_levels=ob_levels)
    obs_dict = obs.to_dict('list')
    keys = obs_dict.keys()
    obs = dict(zip(keys,np.array(list(obs_dict.values()))))
    
    ## Order-flow stylized facts
    interarrival_times = get_interarrival_times(stream_processed)
    trades_distribution = np.array(interarrival_times.groupby(interarrival_times.index.floor('1S')).count())
    # ofs = {"trades_dist":trades_distribution/np.linalg.norm(trades_distribution)}
    ofs = {"trades_dist":trades_distribution}

    # "interarrival_times":np.array(interarrival_times)
    res = {**result, **obs, **ofs}
    
    return res


def get_first_trade(stream_path, start_time = '2021-11-22T09:30:00', end_time = '2021-11-22T10:30:00'):
    
    stream = pd.read_csv(stream_path, header=None, index_col=None)
    
    ## stream cleansing
    stream.drop(6, inplace=True, axis = 1)
    stream.columns=["Time (sec)", "Event Type", "Order ID", "Size", "Price", "Direction"]
    stream['QuoteTime'] = datetime.datetime.fromisoformat('2021-11-22T00:00:00') + pd.to_timedelta(stream.iloc[:,0], 's')
    stream = stream.drop('Time (sec)', axis=1)
    first_column = stream.pop('QuoteTime')
    stream.insert(0, 'QuoteTime', first_column)
    
    start_time = '2021-11-22T09:30:00'
    end_time = '2021-11-22T10:30:00'
    mask = (stream['QuoteTime'] >= start_time) & (stream['QuoteTime'] <= end_time)
    stream = stream.loc[mask]
    stream = stream[(stream['Event Type'] == 4)]
    price = stream.Price.iloc[0]/100
    
    return price
    