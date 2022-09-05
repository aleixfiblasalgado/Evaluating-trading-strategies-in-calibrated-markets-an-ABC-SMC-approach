"""
## Simulator
@author: aleix
"""

import numpy as np
import pandas as pd
import datetime as dt
from dateutil.parser import parse

import warnings
warnings.simplefilter("error", RuntimeWarning)

from Kernel import Kernel
from util import util
from util.order import LimitOrder
from util.oracle.SparseMeanRevertingOracle import SparseMeanRevertingOracle

from agent.ExchangeAgent import ExchangeAgent
from agent.NoiseAgent import NoiseAgent
from agent.ValueAgent import ValueAgent
from agent.market_makers.AdaptiveMarketMakerAgent import AdaptiveMarketMakerAgent
from agent.examples.MomentumAgent import MomentumAgent
from agent.execution.POVExecutionAgent import POVExecutionAgent
from model.LatencyModel import LatencyModel

from realism.realism_utils import get_trades, get_interarrival_times, get_orderbook_stylized_facts

from metrics.minutely_returns import MinutelyReturns
from metrics.aggregation_normality import AggregationNormality
from metrics.autocorrelation import Autocorrelation
from metrics.volatility_clustering import VolatilityClustering

from util.formatting.convert_order_book import process_orderbook
from util.formatting.convert_order_stream import convert_stream_to_format

def rmsc03(parameters, initial_cash = 1000000):
    
    historical_date = '20211122'
    ticker = 'AAPL'
    start_time='09:30:00'
    end_time='10:30:00'
    cash = initial_cash
    seed = 1234

    system_name = "ABIDES: Agent-Based Interactive Discrete Event Simulation"

    print ("=" * len(system_name))
    print (system_name)
    print ("=" * len(system_name))
    print ()

    ## General variables
    verbose = True

    if not seed: seed = int(pd.Timestamp.now().timestamp() * 1000000) % (2 ** 32 - 1)
    np.random.seed(seed)

    util.silent_mode = verbose
    LimitOrder.silent_mode = verbose

    exchange_log_orders = True
    log_orders = None
    book_freq = 0

    simulation_start_time = dt.datetime.now()
    print("Simulation Start Time: {}".format(simulation_start_time))
    print("Configuration seed: {}\n".format(seed))

    ###########################################################################
    ########################## AGENTS CONFIG ##################################

    # historical date to simulate
    historical_date = str(historical_date)
    historical_date = pd.to_datetime(parse(historical_date))
    mkt_open = historical_date + pd.to_timedelta(parse(start_time).strftime('%H:%M:%S'))
    mkt_close = historical_date + pd.to_timedelta(parse(end_time).strftime('%H:%M:%S'))
    agent_count, agents, agent_types = 0, [], []

    # Hyperparameters
    symbol = ticker
    starting_cash = cash ## in cents

    ## Fundamental series hyperparameters
    r_bar = 1e5
    sigma_n = r_bar / 10
    # kappa = 1.67e-16
    # lambda_a = 7e-11
    sigma_s = 0
    # fund_vol = 1e-8
    # megashock_lambda_a = 2.77778e-18
    # megashock_mean = 1e3
    # megashock_var = 5e4

    ## Exchange Agent parameters
    pipeline_delay = 0
    computation_delay = 0

    ## Market maker parameters
    mm_window_size = 'adaptive'
    # mm_pov = 0.025
    mm_num_ticks = 10
    mm_wake_up_freq = '10S'
    mm_min_order_size = 1
    mm_skew_beta = 0
    mm_level_spacing = 5
    # mm_spread_alpha = 0.75
    mm_backstop_quantity = 50000      

    ## Execution agent parameters
    execution_agents = False
    execution_pov = 0.1
    pov_quantity = 12e5 


    ## Oracle

    symbols = {symbol: {'r_bar': r_bar,
                    'kappa': parameters["kappa"],
                    'sigma_s': sigma_s,
                    'fund_vol': parameters["fund_vol"],
                    'megashock_lambda_a': parameters["megashock_lambda_a"],
                    'megashock_mean': parameters["megashock_mean"],
                    'megashock_var': parameters["megashock_var"],
                    'random_state': np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64'))}}

    oracle = SparseMeanRevertingOracle(mkt_open, mkt_close, symbols)


    # 1) Exchange Agent

    #  How many orders in the past to store for transacted volume computation
    # stream_history_length = int(pd.to_timedelta(args.mm_wake_up_freq).total_seconds() * 100)
    stream_history_length = 25000

    agents.extend([ExchangeAgent(id=0,
                             name="EXCHANGE_AGENT",
                             type="ExchangeAgent",
                             mkt_open=mkt_open,
                             mkt_close=mkt_close,
                             symbols=[symbol],
                             log_orders=exchange_log_orders,
                             pipeline_delay=pipeline_delay,
                             computation_delay=computation_delay,
                             stream_history=stream_history_length,
                             book_freq=book_freq,
                             wide_book=True,
                             random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))])
    agent_types.extend("ExchangeAgent")
    agent_count += 1


    # 2) Noise Agents

    num_noise = parameters["num_noise_ag"]
    # These times needed for distribution of arrival times
    noise_mkt_open = historical_date + pd.to_timedelta("09:00:00")  
    noise_mkt_close = historical_date + pd.to_timedelta("16:00:00")
    agents.extend([NoiseAgent(id=j,
                          name="NoiseAgent {}".format(j),
                          type="NoiseAgent",
                          symbol=symbol,
                          starting_cash=starting_cash,
                          wakeup_time=util.get_wake_time(noise_mkt_open, noise_mkt_close),
                          log_orders=log_orders,
                          random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))
               for j in range(agent_count, agent_count + num_noise)])
    agent_count += num_noise
    agent_types.extend(['NoiseAgent'])


    # 3) Value Agents
    num_value = parameters["num_value_ag"]
    agents.extend([ValueAgent(id=j,
                          name="Value Agent {}".format(j),
                          type="ValueAgent",
                          symbol=symbol,
                          starting_cash=starting_cash,
                          sigma_n=sigma_n,
                          r_bar=r_bar,
                          kappa=1.67e-15,
                          lambda_a=parameters["lambda_a"],
                          log_orders=log_orders,
                          random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))
               for j in range(agent_count, agent_count + num_value)])
    agent_count += num_value
    agent_types.extend(['ValueAgent'])



    # 4) Market Maker Agents

    """
    window_size ==  Spread of market maker (in ticks) around the mid price
    pov == Percentage of transacted volume seen in previous `mm_wake_up_freq` that
       the market maker places at each level
       num_ticks == Number of levels to place orders in around the spread
       wake_up_freq == How often the market maker wakes up

    """

    # each elem of mm_params is tuple (window_size, pov, num_ticks, wake_up_freq, min_order_size)
    mm_params = [(mm_window_size, parameters["mm_pov"], mm_num_ticks, mm_wake_up_freq, mm_min_order_size),
                 (mm_window_size, parameters["mm_pov"], mm_num_ticks, mm_wake_up_freq, mm_min_order_size)
                 ]

    num_mm_agents = len(mm_params)
    mm_cancel_limit_delay = 50  # 50 nanoseconds

    agents.extend([AdaptiveMarketMakerAgent(id=j,
                                name="ADAPTIVE_POV_MARKET_MAKER_AGENT_{}".format(j),
                                type='AdaptivePOVMarketMakerAgent',
                                symbol=symbol,
                                starting_cash=starting_cash,
                                pov=mm_params[idx][1],
                                min_order_size=mm_params[idx][4],
                                window_size=mm_params[idx][0],
                                num_ticks=mm_params[idx][2],
                                wake_up_freq=mm_params[idx][3],
                                cancel_limit_delay=mm_cancel_limit_delay,
                                skew_beta=mm_skew_beta,
                                level_spacing=mm_level_spacing,
                                spread_alpha=parameters["mm_spread_alpha"],
                                backstop_quantity=mm_backstop_quantity,
                                log_orders=log_orders,
                                random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                          dtype='uint64')))
               for idx, j in enumerate(range(agent_count, agent_count + num_mm_agents))])
    agent_count += num_mm_agents
    agent_types.extend('POVMarketMakerAgent')

    # 5) Momentum Agents
    num_momentum_agents = parameters["num_momentum_ag"]

    agents.extend([MomentumAgent(id=j,
                             name="MOMENTUM_AGENT_{}".format(j),
                             type="MomentumAgent",
                             symbol=symbol,
                             starting_cash=starting_cash,
                             min_size=1,
                             max_size=10,
                             wake_up_freq='20s',
                             log_orders=log_orders,
                             random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                       dtype='uint64')))
               for j in range(agent_count, agent_count + num_momentum_agents)])
    agent_count += num_momentum_agents
    agent_types.extend("MomentumAgent")

    # 6) Execution Agent

    trade = True if execution_agents else False

    pov_agent_start_time = mkt_open + pd.to_timedelta('00:30:00')
    pov_agent_end_time = mkt_close - pd.to_timedelta('00:30:00')
    pov_proportion_of_volume = execution_pov
    pov_frequency = '1min'
    pov_direction = "BUY"

    pov_agent = POVExecutionAgent(id=agent_count,
                              name='POV_EXECUTION_AGENT',
                              type='ExecutionAgent',
                              symbol=symbol,
                              starting_cash=starting_cash,
                              start_time=pov_agent_start_time,
                              end_time=pov_agent_end_time,
                              freq=pov_frequency,
                              lookback_period=pov_frequency,
                              pov=pov_proportion_of_volume,
                              direction=pov_direction,
                              quantity=pov_quantity,
                              trade=trade,
                              log_orders=log_orders,  # needed for plots so conflicts with others
                              random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                          dtype='uint64')))
    execution_agents = [pov_agent]
    agents.extend(execution_agents)
    agent_types.extend("ExecutionAgent")
    agent_count += 1

    ###########################################################################
    ################### KERNEL AND OTHER CONFIG ###############################

    kernel = Kernel("RMSC03 Kernel", parameters = parameters, 
                    random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))

    kernelStartTime = historical_date
    kernelStopTime = mkt_close + pd.to_timedelta('00:01:00')

    defaultComputationDelay = 50  # 50 nanoseconds

    # LATENCY

    latency_rstate = np.random.RandomState(seed=np.random.randint(low=0, high=2**32))

    # All agents sit on line from Seattle to NYC
    nyc_to_seattle_meters = 3866660
    pairwise_distances = util.generate_uniform_random_pairwise_dist_on_line(0.0, nyc_to_seattle_meters, agent_count,
                                                                        random_state=latency_rstate)
    pairwise_latencies = util.meters_to_light_ns(pairwise_distances)

    model_args = {
        'connected': True,
        'min_latency': pairwise_latencies
    }

    latency_model = LatencyModel(latency_model='deterministic',
                             random_state=latency_rstate,
                             kwargs=model_args)

    # KERNEL

    print("Model parameters: ", parameters)
    
    try:
        _, stream_df, orderbook_df = kernel.runner(agents=agents,
                      startTime=kernelStartTime,
                      stopTime=kernelStopTime,
                      agentLatencyModel=latency_model,
                      defaultComputationDelay=defaultComputationDelay,
                      oracle=oracle)
        
        simulation_end_time = dt.datetime.now()
        print("Simulation End Time: {}".format(simulation_end_time))
        print("Time taken to run simulation: {}".format(simulation_end_time - simulation_start_time))

        ## Data processing
        ohlcv = get_trades(stream_df)
        stream_processed = convert_stream_to_format(stream_df.reset_index(), fmt='plot-scripts')
        stream_processed = stream_processed.set_index('TIMESTAMP')
        num_levels=3
        processed_ob = process_orderbook(orderbook_df, num_levels)

        bust = sum( (ohlcv.iloc[:,0:4]<0).any() ) > 0

        ## Asset returns stylized facts
        all_metrics = [MinutelyReturns, AggregationNormality, Autocorrelation, 
                       VolatilityClustering]
        metric_names = ["MinutelyReturns", "AggregationNormality", "Autocorrelation", 
                        "VolatilityClustering"]    
        result = dict()

        if bust:
            for my_metric, metric_name in zip(all_metrics, metric_names):
                my_metric = my_metric()
                result[metric_name] = np.empty(0, dtype=float, order='C')
        else: 
            for my_metric, metric_name in zip(all_metrics, metric_names):
                my_metric = my_metric()
                result[metric_name] = np.array(my_metric.compute(ohlcv))

            result.update({'VolatilityClustering':result['VolatilityClustering'].flatten()})

        ## Volume stylized facts
        obs = get_orderbook_stylized_facts(orderbook_df, processed_ob, stream_processed, num_levels)
        obs_dict = obs.to_dict('list')
        keys = obs_dict.keys()
        obs = dict(zip(keys,np.array(list(obs_dict.values()))))

        ## Order-flow stylized facts
        interarrival_times = get_interarrival_times(stream_processed)
        trades_distribution = interarrival_times.groupby(interarrival_times.index.floor('1S')).count()
        ofs = {"trades_dist":np.array(trades_distribution)}

        # "interarrival_times":np.array(interarrival_times)
        res = {**result, **obs, **ofs}

        return(res)
        
        
    except (RuntimeWarning, MemoryError):
        print("Parameters leading to an unfeasible market scenario...")
        return {
            "MinutelyReturns": np.empty(0, dtype=float, order='C'),
            "AggregationNormality": np.empty(0, dtype=float, order='C'),
            "Autocorrelation": np.empty(0, dtype=float, order='C'),
            "VolatilityClustering": np.empty(0, dtype=float, order='C'),
            "spread": np.empty(0, dtype=float, order='C'),
            "volume_t": np.empty(0, dtype=float, order='C'),
            "imbalance": np.empty(0, dtype=float, order='C'),
            "trades_dist": np.empty(0, dtype=float, order='C')
            }