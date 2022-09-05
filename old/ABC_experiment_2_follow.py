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
import json

from market_simulations import AAPL_10 as market
from plotting_utils import plot_history
from market_data import get_market_data

if __name__ == "__main__":

    prior = pyabc.Distribution(
        num_value_ag=pyabc.RV('randint', 20, 800),
        num_momentum_ag=pyabc.RV('randint', 20, 800),
        num_noise_ag=pyabc.RV('randint', 10, 8000),
        mm_pov=pyabc.RV("uniform", 0.0001, 0.5),
        kappa=pyabc.RV("uniform", 1e-17, 1e-15),
        fund_vol=pyabc.RV("uniform", 1e-10, 1e-6),
        megashock_lambda_a=pyabc.RV("uniform", 1e-19, 1e-17),
        megashock_mean=pyabc.RV("uniform", 500, 3000),
        megashock_var=pyabc.RV("uniform", 20000, 60000),
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
        'mm_pov':pyabc.MultivariateNormalTransition(scaling=0.3),
        'kappa':pyabc.MultivariateNormalTransition(scaling=0.3),
        'fund_vol':pyabc.MultivariateNormalTransition(scaling=0.3),
        'megashock_lambda_a':pyabc.MultivariateNormalTransition(scaling=0.3),
        'megashock_mean':pyabc.MultivariateNormalTransition(scaling=0.3),
        'megashock_var':pyabc.MultivariateNormalTransition(scaling=0.3),
        }
    )
    
    pop_size = 150
    epsilon = 1e-9
    db_path = os.path.join(os.getcwd(), "MSc_thesis/ABCSMC_Calibration/ABC_output")
    log_file = tempfile.mkstemp()[1]
    sample_log_file = tempfile.mkstemp()[1]

    distance.scale_log_file = f"{log_file}_{'Aggregated_distance'}.json"
    distance.info_sample_log_file = f"{sample_log_file}_{'Aggregated_distance'}"
    
    abc_continued = pyabc.ABCSMC(market, prior, distance,  population_size=pop_size, 
                       transitions=transition, max_nr_recorded_particles=int(2e4), 
                       sampler=pyabc.sampler.MulticoreEvalParallelSampler(n_procs=128))
    
    abc_continued.load("sqlite:///"+db_path+"/experiment_2.db", 1)
    
    abc_continued.run(minimum_epsilon=epsilon, max_nr_populations=2)