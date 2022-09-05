from agent.TradingAgent import TradingAgent
from message.Message import Message
from util.order.LimitOrder import LimitOrder
from util.order.MarketOrder import MarketOrder
from util.util import log_print

from copy import deepcopy
import sys
import numpy as np
import pandas as pd


class MomentumAgent(TradingAgent):
    def __init__ (self, id, name, symbol, startingCash=100000, lookback=50):
        # Base class init.
        super().__init__(id, name, startingCash)
        
        self.symbol = symbol
        self.lookback = lookback
        self.state = "AWAITING_WAKEUP"
        
        self.trades = []
        
    def wakeup (self, currentTime):
        can_trade = super().wakeup(currentTime)
        
        if not can_trade : return
        
        self.getLastTrade (self.symbol)
        self.state = "AWAITING_LAST_TRADE"
    
    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)
        
        if self.state == "AWAITING_LAST_TRADE" and \
            msg.type == "QUERY_LAST_TRADE":
            
            last = self.last_trade[self.symbol]
            self.trades = (self.trades + [last])[:self.lookback]
            
            if len(self.trades) >= self.lookback:
                m, b = np.polyfit(range(len(self.trades)),
                                  self.trades, 1)
                pred = self.lookback * m + b
                
                holdings = self.getHoldings(self.symbol)
                
                if pred > last:
                    self.placeLimitOrder(self.symbol, 100-holdings, True, self.MKT_BUY)
                else:
                    self.placeLimitOrder(self.symbol, 100+holdings, True, self.MKT_SELL)
            
            self.setWakeup(currentTime + pd.Timedelta("1m"))
            self.state = "AWAITING_WAKEUP"