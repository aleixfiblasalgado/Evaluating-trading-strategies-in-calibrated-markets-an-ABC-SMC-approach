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
import datetime as dt
import scipy as sp

from market_simulations import rmsc03
from plotting_utils import plot_history

if __name__ == "__main__":
    
    params_true={"num_value_ag":100 , "num_momentum_ag":25, "num_noise_ag":5000, 
                  "megashock_lambda_a":2.77778e-18,
                  "kappa":1.67e-16,
                  "fund_vol":1e-8,
                  "mm_pov":0.025,
                  "mm_spread_alpha":0.75,
                  "lambda_a": 7e-11,
                  "megashock_mean":1e3,
                  "megashock_var":5e4
                  }

    data = rmsc03(params_true)

    prior = pyabc.Distribution(
        num_value_ag=pyabc.RV('randint', 20, 800),
        num_momentum_ag=pyabc.RV('randint', 20, 800),
        num_noise_ag=pyabc.RV('randint', 20, 8000),
        megashock_lambda_a=pyabc.RV("uniform", 1e-18, 1e-10),
        kappa=pyabc.RV("uniform", 0, 1e-2),
        fund_vol=pyabc.RV("uniform", 0, 0.1),
        mm_pov=pyabc.RV("uniform", 0.01, 0.5),
        mm_spread_alpha=pyabc.RV("uniform", 0.5, 1),
        lambda_a=pyabc.RV("uniform", 1e-11, 1e-10),
        megashock_mean=pyabc.RV("norm", 3000, 500),
        megashock_var=pyabc.RV("uniform", 10000, 80000)
    )
    
    def distance_returns(x, x_0):
        try:
            ks = sp.stats.ks_2samp(x['MinutelyReturns'], x_0['MinutelyReturns'])
            dist = ks.statistic
        except:
            dist = 9999
            print("Stock goes bust")
        return dist
    
    def distance_aggnorm(x, x0):
        try:
            ks = sp.stats.ks_2samp(x['AggregationNormality'], x0['AggregationNormality'])
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

    def distance_spread(x, x0):
        try:
            ks = sp.stats.ks_2samp(x['spread'], x0['spread'])
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
        distance_aggnorm,
        distance_autocorr,
        distance_vol_clust,
        distance_spread,
        distance_vol,
        distance_imb,
        distance_trades_dist
        ], adaptive=True)
    
    transition = pyabc.AggregatedTransition(
    mapping={
        'num_value_ag': pyabc.DiscreteJumpTransition(
            domain=np.arange(20, 801), p_stay=0.7
        ),
        'num_momentum_ag': pyabc.DiscreteJumpTransition(
            domain=np.arange(20, 801), p_stay=0.7
        ),
        'num_noise_ag': pyabc.DiscreteJumpTransition(
            domain=np.arange(20, 8001), p_stay=0.7
        ),
        'megashock_lambda_a':pyabc.MultivariateNormalTransition(),
        'kappa':pyabc.MultivariateNormalTransition(),
        'fund_vol':pyabc.MultivariateNormalTransition(),
        'mm_pov':pyabc.MultivariateNormalTransition(),
        'mm_spread_alpha':pyabc.MultivariateNormalTransition(),
        'lambda_a':pyabc.MultivariateNormalTransition(),
        'megashock_mean':pyabc.MultivariateNormalTransition(),
        'megashock_var':pyabc.MultivariateNormalTransition(),
        }
    )
    
    pop_size = 50
    abc = pyabc.ABCSMC(rmsc03, prior, distance,  population_size=pop_size, 
                       transitions=transition, max_nr_recorded_particles = 2000)

    abc.new(pyabc.create_sqlite_db_id(dir_= os.getcwd(), file_='proof_of_concept.db'), data)
    
    epsilon = 0.001
    
    history1 = abc.run(max_nr_populations=6, minimum_epsilon=epsilon)
    
    # plot_history(history1, params_true)
    
    # abc_continued = pyabc.ABCSMC(rmsc03, prior, distance,  population_size=pop_size)
    # abc_continued.load("sqlite:///"+db_path, 1)
    
    # history1 = abc_continued.history
    
    # plot_history(history1, params_true)
    
    # pyabc.visualization.plot_sample_numbers(history1)