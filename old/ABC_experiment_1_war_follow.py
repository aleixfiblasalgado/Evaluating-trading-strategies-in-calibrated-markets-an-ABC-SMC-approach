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

    prior = pyabc.Distribution(
        num_value_ag=pyabc.RV('randint', 20, 800),
        num_momentum_ag=pyabc.RV('randint', 20, 800),
        num_noise_ag=pyabc.RV('randint', 10, 8000),
        mm_pov=pyabc.RV("uniform", 0.0001, 0.5),
    )
    
    class returnsSumstat(pyabc.Sumstat):

        def __call__(self, data: dict) -> np.ndarray:
            
            ret_stat = np.stack((data['MinutelyReturns'], 
                    data['Autocorrelation']), axis = 1)
                
            if ret_stat.shape == (59, 2):
                print("Feasible scenario")
            else:
                print("Unfeasible scenario")
                ret_stat = np.zeros((59, 2))
                
            return ret_stat
    
    class volclusSumstat(pyabc.Sumstat):

        def __call__(self, data: dict) -> np.ndarray:
            
            vc_stat = data['VolatilityClustering'].reshape(-1, 1)
                
            if vc_stat.shape == (50, 1):
                print("Feasible scenario")
            else:
                print("Unfeasible scenario")
                vc_stat = np.zeros((50, 1))
 
            return vc_stat
        
        
    class order_flow_Sumstat(pyabc.Sumstat):

        def __call__(self, data: dict) -> np.ndarray:

            of_stat = np.stack((data['volume_t'], data['imbalance']), axis = 1)
        
            if of_stat.shape == (180, 2):
                print("Feasible scenario")
            else:
                print("Unfeasible scenario")
                of_stat = np.zeros((180, 2))
                
            return of_stat

    class IdSumstat(pyabc.Sumstat):

        def __call__(self, data: dict) -> np.ndarray:
            
            vol_stat = data['trades_dist'].reshape(-1, 1)
            
            if vol_stat.shape == (3600, 1):
                print("Feasible scenario")
            else:
                print("Unfeasible scenario")
                vol_stat = np.zeros((3600, 1))
 
            return vol_stat
    
    distance = pyabc.AdaptiveAggregatedDistance([
        pyabc.SlicedWassersteinDistance(
            metric="sqeuclidean",
            p=2,
            sumstat=returnsSumstat(),
            n_proj=10,
        ),
        pyabc.SlicedWassersteinDistance(
            metric="sqeuclidean",
            p=2,
            sumstat=volclusSumstat(),
            n_proj=10,
        ),
        pyabc.SlicedWassersteinDistance(
            metric="sqeuclidean",
            p=2,
            sumstat=order_flow_Sumstat(),
            n_proj=10,
        ),
        pyabc.SlicedWassersteinDistance(
            metric="sqeuclidean",
            p=2,
            sumstat=IdSumstat(),
            n_proj=10,
        )
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
    epsilon = 1e-8
    
    db_path = os.path.join(os.getcwd(), "MSc_thesis/ABCSMC_Calibration/ABC_output")
    log_file = tempfile.mkstemp()[1]
    sample_log_file = tempfile.mkstemp()[1]

    distance.scale_log_file = f"{log_file}_{'Aggregated_distance'}.json"
    distance.info_sample_log_file = f"{sample_log_file}_{'Aggregated_distance'}"
    
    abc_continued = pyabc.ABCSMC(market, prior, distance,  population_size=pop_size, 
                                 transitions=transition, max_nr_recorded_particles=int(2e4))
    
    abc_continued.load("sqlite:///"+db_path+"/experiment_1_war.db", 1)
    
    abc_continued.run(minimum_epsilon=epsilon, max_nr_populations=1)