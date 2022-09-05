"""
## Simulator
@author: aleix
"""

import numpy as np

import warnings
warnings.simplefilter("error", RuntimeWarning)

from market_simulations import rmsc03_4 as syntehtic_market
from market_simulations import AAPL_10 as real_market

import signal


def rmsc03_4_t(parameters):
    
    def signal_handler(signum, frame):
        raise TimeoutError("Timed out!")
        
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(600)   # one hour
    try:
        ss = syntehtic_market(parameters)
    except TimeoutError:
        print("Timed out!")
        ss = {
            "MinutelyReturns": np.empty(0, dtype=float, order='C'),
            "Autocorrelation": np.empty(0, dtype=float, order='C'),
            "VolatilityClustering": np.empty(0, dtype=float, order='C'),
            "volume_t": np.empty(0, dtype=float, order='C'),
            "imbalance": np.empty(0, dtype=float, order='C'),
            "trades_dist": np.empty(0, dtype=float, order='C')
            }
    finally:
        signal.alarm(0)
        
    return(ss)
        
def AAPL_10(parameters):
    
    def signal_handler(signum, frame):
        raise TimeoutError("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(600)   # one hour
    try:
        ss= real_market(parameters)
    except TimeoutError:
        print("Timed out!")
        ss = {
            "MinutelyReturns": np.empty(0, dtype=float, order='C'),
            "Autocorrelation": np.empty(0, dtype=float, order='C'),
            "VolatilityClustering": np.empty(0, dtype=float, order='C'),
            "volume_t": np.empty(0, dtype=float, order='C'),
            "imbalance": np.empty(0, dtype=float, order='C'),
            "trades_dist": np.empty(0, dtype=float, order='C')
            }
    finally:
        signal.alarm(0)
        
    return(ss)