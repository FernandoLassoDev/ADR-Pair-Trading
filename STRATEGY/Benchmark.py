import pandas as pd
import numpy as np
import sys
import random
from statsmodels.regression.rolling import RollingOLS
from STRATEGY.BaseTradeEngine import BaseTradeEngine

def get_src_cls(source_name):
    return getattr(sys.modules[__name__], source_name)

class Benchmark(BaseTradeEngine):
    
    def __init__(self, *args, **kwargs):
        super(Benchmark, self).__init__(*args, **kwargs)
        
    # Run a backtest with given parameter inputs
    def process(self,  entryZ = 2, exitZ = 0, lag = 0, rounding = 3, resample = 1,
                 windowZ = 30, windowOLS = 150, train_rng = [0,1], **kwargs):
       
        minWindowOLS = int(min(max(windowOLS/np.sqrt(resample),windowZ),len(self.original_x)/2))
        
        # Calibrate OLS if necessary
        if self.windowOLS != minWindowOLS or resample != self.resample:
            self.resampling(resample)
            self.calibrate(minWindowOLS)
            self.resample = resample
        
        # Get start and end time for data
        start_hist, end_hist = self.get_indices(train_rng)
        
        #In any case, start data after enough data gathered for the OLS window
        min_start = (minWindowOLS-1) / len(self.x) 
        train_rng[0] = max(train_rng[0],min_start)
        
        subsamples = [self.buy_x , self.buy_y, self.sell_x, self.sell_y, self.beta,  self.timestamp]
        
        [buy_x, buy_y, sell_x, sell_y, beta, time]  = self.get_sample(start_hist, end_hist, *subsamples)
        
        parameters = {'beta': beta, 'entryZ':entryZ , 'exitZ': exitZ, 
                      'lag': lag, 'rounding': rounding, 'windowZ': windowZ}
        
        parameters.update(kwargs)
        
        # Create the record
        self.record = self.backtest(buy_x, buy_y, sell_x, sell_y, time, **parameters)
        
        # Legacy
        self.reward = sum(self.record.port_rets)
        
    # Use a Rolling OLS with given window size to calculate the hedge ratio
    def calibrate(self, windowOLS, **kwargs):
        
        #x, y, time = super().get_sample(self.x,self.y, self.timestamp, start_hist, end_hist)
        #model = RollingOLS(endog =self.y, exog=self.x,window=self.windowOLS)
        #rres = model.fit()
        #self.beta = rres.params.reshape(-1, )
        self.windowOLS = windowOLS
        
        df = pd.DataFrame({'y':self.y,'x':self.x,'c':1})
        
        model = RollingOLS(endog =df['y'], exog=df[['x']],window=self.windowOLS)
        rres = model.fit()
      
        self.beta = rres.params['x'].values.reshape(-1, )
    
    def calculate_hr(self, record, beta, rounding, **kwargs):
        record['hr'] = beta.round(int(rounding))
        return record
        
        
    def calculate_signals(self, record, windowZ, lag, **kwargs):
        
        buy_spread  = record.buy_spread
        sell_spread = record.sell_spread
        
        windowZ= int(windowZ)
        # Rolling mean and standard deviation calculations for a given historical window
        meanBuySpread  = buy_spread.rolling(window=windowZ).mean()
        stdBuySpread   = buy_spread.rolling(window=windowZ).std()
        meanSellSpread = sell_spread.rolling(window=windowZ).mean()
        stdSellSpread  = sell_spread.rolling(window=windowZ).std()
        
        record['buySignal']  = ((meanSellSpread - buy_spread )/stdBuySpread ).shift(int(lag))
        record['sellSignal'] = ((-meanBuySpread + sell_spread)/stdSellSpread).shift(int(lag))
       
        return record
    
    def calculate_thresholds(self, record, entryZ, exitZ, **kwargs):
        
        record['entryThreshold'] = entryZ
        record['exitThreshold']  = exitZ
        
        return record
    
   
        
