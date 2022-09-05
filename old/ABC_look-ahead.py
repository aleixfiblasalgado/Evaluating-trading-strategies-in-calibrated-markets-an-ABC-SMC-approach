"""
## Simulator
@author: aleix
"""
import logging
import os
import tempfile
import time

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
    
    #params_true={"num_value_ag":10 , "num_momentum_ag":56, "num_noise_ag":5767}
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
        num_value_ag=pyabc.RV('randint', 10, 500),
        num_momentum_ag=pyabc.RV('randint', 0, 100),
        num_noise_ag=pyabc.RV('randint', 0, 8000),
        megashock_lambda_a=pyabc.RV("uniform",1e-19 , 1e-18),
        kappa=pyabc.RV("uniform", 0, 1),
        fund_vol=pyabc.RV("uniform", 0, 1),
        mm_pov=pyabc.RV("uniform", 0, 1),
        mm_spread_alpha=pyabc.RV("uniform", 0, 1),
        lambda_a=pyabc.RV("uniform", 1e-14, 1e-11),
        megashock_mean=pyabc.RV("norm", 2500, 500),
        megashock_var=pyabc.RV("uniform", 0, 10000)
    )
    
    def distance_returns(x, x_0):
        ks = sp.stats.ks_2samp(x["MinutelyReturns"], x_0["MinutelyReturns"])
        return ks.statistic
    
    def distance_aggnorm(x, x0):
        ks = sp.stats.ks_2samp(x["AggregationNormality"], x0["AggregationNormality"])
        return ks.statistic
        
    def distance_autocorr(x, x0):
        ks = sp.stats.ks_2samp(x["Autocorrelation"], x0["Autocorrelation"])
        return ks.statistic

    def distance_vol_clust(x, x0):
        ks = sp.stats.ks_2samp(x["VolatilityClustering"], x0["VolatilityClustering"])
        return ks.statistic
    
    def distance_spread(x, x0):
        ks = sp.stats.ks_2samp(x["spread"], x0["spread"])
        return ks.statistic
    
    def distance_vol(x, x0):
        ks = sp.stats.ks_2samp(x["volume_t"], x0["volume_t"])
        return ks.statistic
    
    def distance_imb(x, x0):
        ks = sp.stats.ks_2samp(x["imbalance"], x0["imbalance"])
        return ks.statistic
    
    def distance_int_t(x, x0):
        ks = sp.stats.ks_2samp(x["interarrival_times"], x0["interarrival_times"])
        return ks.statistic
    
    def distance_trades_dist(x, x0):
        ks = sp.stats.ks_2samp(x["trades_dist"], x0["trades_dist"])
        return ks.statistic
    
    distance = pyabc.AdaptiveAggregatedDistance([
        distance_returns,
        distance_aggnorm,
        distance_autocorr,
        distance_vol_clust,
        distance_spread,
        distance_vol,
        distance_imb,
        distance_int_t,
        distance_trades_dist
        ], adaptive=True)
    
    transition = pyabc.AggregatedTransition(
    mapping={
        'num_value_ag': pyabc.DiscreteJumpTransition(
            domain=np.arange(501), p_stay=0.6
        ),
        'num_momentum_ag': pyabc.DiscreteJumpTransition(
            domain=np.arange(101), p_stay=0.6
        ),
        'num_noise_ag': pyabc.DiscreteJumpTransition(
            domain=np.arange(8001), p_stay=0.7
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
    
    pop_size = 100
    
    hs_la = []
    sampler_logfiles = []
    niter = 10
    db_path = os.path.join(os.getcwd()+'/ABC_output', "proof_concept_LA.db")
    
    for i in range(0, niter):
        logfile = tempfile.mkstemp(prefix="redis_log", suffix=".csv")[1]
        sampler_logfiles.append(logfile)
        redis_sampler = pyabc.sampler.RedisEvalParallelSamplerServerStarter(
            # main field: in generation t already preemptively sample for t+1 if cores
            #  are available
            look_ahead=True,
            # whether to delay evaluation until the next generation has really
            #  started, this is necessary if any component s.a. eps, distance is
            #  adaptive
            look_ahead_delay_evaluation=True,
            # determines how many samples to sample preemptively maximally without
            #  checking
            max_n_eval_look_ahead_factor=2,
            # a file for some sampler debugging output
            log_file=logfile,
            workers=pyabc.nr_cores_available(),
        )
        
        abc = pyabc.ABCSMC(rmsc03, prior, distance,  population_size=pop_size, 
                           transitions=transition, eps = 0.1, sampler = redis_sampler)
        
        abc.new("sqlite:///" + db_path, data)
        
        h = abc.run(max_nr_populations=6)
        
        hs_la.append(h)
        
    # plot_history(history1, params_true)
    
    # abc_continued = pyabc.ABCSMC(rmsc03, prior, distance,  population_size=pop_size)
    # abc_continued.load("sqlite:///"+db_path, 9)
    
    # history1 = abc_continued.history
    
    # plot_history(history1, params_true)