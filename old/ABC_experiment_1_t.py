"""
## Simulator
@author: aleix
"""
import logging

import pyabc
pyabc.settings.set_figure_params('pyabc')
# for debugging
for logger in ["ABC.Distance", "ABC.Predictor", "ABC.Sumstat"]:
    logging.getLogger(logger).setLevel(logging.DEBUG)

import numpy as np
import os
import tempfile
import datetime as dt
import scipy as sp

from market_simulations import rmsc03_4 as market
from plotting_utils import plot_history

if __name__ == "__main__":
    
    params_true={"num_value_ag":100 , "num_momentum_ag":25, "num_noise_ag":5000, 
                  "mm_pov":0.025}
    
    data = market(params_true)

    prior = pyabc.Distribution(
        num_value_ag=pyabc.RV('randint', 20, 800),
        num_momentum_ag=pyabc.RV('randint', 20, 800),
        num_noise_ag=pyabc.RV('randint', 10, 8000),
        mm_pov=pyabc.RV("uniform", 0.0001, 0.5),
    )
    
    
    def distance_returns(x, x_0):
        try:
            ks = sp.stats.ks_2samp(x['MinutelyReturns'], x_0['MinutelyReturns'])
            dist = ks.statistic
        except:
            dist = 9999
            print("Stock goes bust")
        return dist
        
    def distance_autocorr(x, x0):
        try:
            ks = sp.stats.ks_2samp(x['Autocorrelation'], x0['Autocorrelation'])
            dist = ks.statistic
        except:
            dist = 9999
            print("Stock goes bust")
        return dist

    def distance_vol_clust(x, x0):
        try:
            ks = sp.stats.ks_2samp(x['VolatilityClustering'], x0['VolatilityClustering'])
            dist = ks.statistic
        except:
            dist = 9999
            print("Stock goes bust")
        return dist
    
    def distance_vol(x, x0):
        try:
            ks = sp.stats.ks_2samp(x['volume_t'], x0['volume_t'])
            dist = ks.statistic
        except:
            dist = 9999
            print("Stock goes bust")
        return dist
    
    def distance_imb(x, x0):
        try:
            ks = sp.stats.ks_2samp(x['imbalance'], x0['imbalance'])
            dist = ks.statistic
        except:
            dist = 9999
            print("Stock goes bust")
        return dist
    
    def distance_trades_dist(x, x0):
        try:
            ks = sp.stats.ks_2samp(x['trades_dist'], x0['trades_dist'])
            dist = ks.statistic
        except:
            dist = 9999
            print("Stock goes bust")
        return dist
    
    distance = pyabc.AdaptiveAggregatedDistance([
        distance_returns,
        distance_autocorr,
        distance_vol_clust,
        distance_vol,
        distance_imb,
        distance_trades_dist
        ], adaptive=True)
    
    transition = pyabc.AggregatedTransition(
    mapping={
        'num_value_ag': pyabc.DiscreteJumpTransition(
            domain=np.arange(20, 801), p_stay=0.3
        ),
        'num_momentum_ag': pyabc.DiscreteJumpTransition(
            domain=np.arange(20, 801), p_stay=0.3
        ),
        'num_noise_ag': pyabc.DiscreteJumpTransition(
            domain=np.arange(10, 8001), p_stay=0.3
        ),
        'mm_pov':pyabc.MultivariateNormalTransition(),
        }
    )
    
    pop_size = 100
    abc = pyabc.ABCSMC(market, prior, distance,  population_size=pop_size, 
                       transitions=transition, max_nr_recorded_particles=int(2e4))

    db_path = os.path.join(os.getcwd(), "MSc_thesis/ABCSMC_Calibration/ABC_output")

    abc.new(pyabc.create_sqlite_db_id(dir_= db_path, file_='experiment_1_t.db'), data)
    
    epsilon = 1e-8
    
    history1 = abc.run(max_nr_populations=2, minimum_epsilon=epsilon)