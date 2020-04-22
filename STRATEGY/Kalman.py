import pandas as pd
import numpy as np
import sys
import random
from pykalman import KalmanFilter
from STRATEGY.BaseTradeEngine import BaseTradeEngine
import statsmodels.api as sm

def get_src_cls(source_name):
    return getattr(sys.modules[__name__], source_name)

class Kalman(BaseTradeEngine):
    
    def __init__(self, *args, **kwargs):
        super(Kalman, self).__init__(*args, **kwargs)
        
    # Run a backtest with given parameter inputs
    def process(self, rounding = 3, lag = 0, resample = 1, train_rng = [0,1], **kwargs):
        
        # Recalculate state means if necessary
        if resample != self.resample:
            self.resampling(resample)
            self.calibrate()  
            self.resample = resample
            
        # Get start and end time for bids and asks
        start_hist, end_hist = self.get_indices(train_rng)
        
        subsamples = [self.x, self.y, self.buy_x , self.buy_y, self.sell_x, self.sell_y, 
                     self.hr, self.std_q, self.intercept,  self.timestamp]
        
        [x, y, buy_x, buy_y, sell_x, sell_y, hr, std_q, intercept, time] = \
                                        self.get_sample(start_hist, end_hist, *subsamples)
                                           
        
        parameters = {'hr': hr, 'std_q':std_q, 'intercept':intercept, 
                      'lag': lag, 'rounding': rounding}
        
        parameters.update(kwargs)
        
        # Create the record
        self.record = self.backtest(buy_x, buy_y, sell_x, sell_y, time, **parameters) 
        
        # Legacy
        self.reward = sum(self.record.port_rets)
        
       
    def calibrate(self, **kwargs):
        
        a = self.KalmanFilterAverage(self.x,self.timestamp)
        b = self.KalmanFilterAverage(self.y,self.timestamp)
        #a=pd.Series(self.x)
        #b=pd.Series(self.y)
        self.hr, self.intercept, self.std_q = self.KalmanFilterRegression(a,b)
      
    
    def calculate_hr(self, record, hr, intercept, std_q, rounding, **kwargs):
        record['hr'] = hr.round(int(rounding))
        record['intercept'] = intercept
        record['std_q'] = std_q
        return record
        
    def calculate_signals(self, record, lag, **kwargs):
        
        record['ask_pred'] = record.intercept + np.multiply(record.buy_x , record.hr)
        record['bid_pred'] = record.intercept + np.multiply(record.sell_x, record.hr)
       
        ask_error = record.buy_y  - record.ask_pred  
        bid_error = record.sell_y - record.bid_pred 
       
        record['buySignal']  = -ask_error.shift(int(lag))
        record['sellSignal'] = +bid_error.shift(int(lag))
       
        return record
    
    def calculate_thresholds(self, record, entryMult, exitMult, **kwargs):
         
        record['entryThreshold'] = entryMult * record.std_q
        record['exitThreshold']  = exitMult * record.std_q
        
        return record
        
        
    @staticmethod
    def KalmanFilterAverage(x, t):
      # Construct a Kalman filter
        kf = KalmanFilter(transition_matrices = [1],
         observation_matrices = [1],
         initial_state_mean = 0,
         initial_state_covariance = 1,
         observation_covariance=1,
         transition_covariance=.01)

        # Use the observed values of the price to get a rolling mean
        state_means, _ = kf.filter(x)
        state_means = pd.Series(state_means.flatten(), index=t)
        return state_means

    # Kalman filter regression
    @staticmethod
    def KalmanFilterRegression(x,y):
        delta = 1e-3
        trans_cov = delta / (1 - delta) * np.eye(2) # How much random walk wiggles
        obs_mat = np.expand_dims(np.vstack([[x], [np.ones(len(x))]]).T, axis=1)

        kf = KalmanFilter(n_dim_obs=1, n_dim_state=2, # y is 1-dimensional, (alpha, beta) is 2-dimensional
        initial_state_mean=[0,0],
        initial_state_covariance=np.ones((2, 2)),
        transition_matrices=np.eye(2),
        observation_matrices=obs_mat,
        observation_covariance=2,
        transition_covariance=trans_cov)
        # Use the observations y to get running estimates and errors for the state parameters
        state_means, state_covs = kf.filter(y.values)
        slope = state_means[:, 0] 
        intercept = state_means[:, 1] 
        std_q = np.array([np.sqrt(cov[0][0]) for cov in state_covs])
        return slope, intercept, std_q

    
    
