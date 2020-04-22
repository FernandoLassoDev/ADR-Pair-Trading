import pandas as pd
import numpy as np
import sys
import random
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
import statsmodels.api as sm

def get_src_cls(source_name):
    return getattr(sys.modules[__name__], source_name)

class Kalman():

    def __init__(self, x, y, on, col_name, is_cleaned=True, rm_outliers = False, state_means = None):
        if is_cleaned is not True:
            x, y = Kalman.clean_data(x, y, on, col_name, rm_outliers)
        self.timestamp = x[on].values
        self.x = x[col_name].values.reshape(-1, )
        self.y = y[col_name].values.reshape(-1, )
        self.halflife = 0
        
        if (state_means is None):
            self.calibrate(0, len(x))  
        else:
            self.state_means = state_means
        
        self._reward = 0
        self._record = None
        
    @property
    def reward(self):
        return self._reward

    @reward.setter
    def reward(self, value):
        self._reward = value

    @property
    def record(self):
        return self._record

    @record.setter
    def record(self, value):
        self._record = value

    @classmethod
    def clean_data(cls, x, y, on, col_name, rm_outliers):
        
     
        x.replace([np.inf, -np.inf], np.nan, inplace=True)
        y.replace([np.inf, -np.inf], np.nan, inplace=True)
        merged_df = pd.merge(left=x, right=y, on=on, how='outer')
        clean_df  = merged_df.loc[merged_df.notnull().all(axis=1), :]
    
        # Remove outliers that are more than 2 standard deviations away from the minimum of the
        # rolling mean of 5 observations later or 5 observations before
        if rm_outliers:
            clean_df['remove'] = False
            for c in ['x','y']:
                clean_df['dif_'+c+'_before'] = abs(clean_df[col_name + '_'+c] - \
                                                   clean_df[col_name + '_'+c].rolling(5).mean().shift(1))
                clean_df['dif_'+c+'_after'] = abs(clean_df[col_name + '_'+c] - \
                                                  clean_df[col_name + '_'+c].iloc[::-1].rolling(5).mean().shift(1).iloc[::-1])
                clean_df['sd_'+c+'_before'] = clean_df[col_name + '_'+c].rolling(25).std().shift(1)
                clean_df['sd_'+c+'_after'] = clean_df[col_name + '_'+c].iloc[::-1].rolling(25).std().shift(1).iloc[::-1]

                clean_df['use_before'+c] = pd.DataFrame([clean_df['dif_'+c+'_before'],clean_df['dif_'+c+'_after']]).min() \
                                                    == clean_df['dif_'+c+'_before']
                
                clean_df['sd_'+c] = clean_df['sd_'+c+'_after']
                clean_df['dif_'+c] = clean_df['dif_'+c+'_after']
                clean_df['sd_'+c][clean_df['use_before'+c]] = clean_df['sd_'+c+'_before'][clean_df['use_before'+c]]
                clean_df['dif_'+c][clean_df['use_before'+c]] = clean_df['dif_'+c+'_before'][clean_df['use_before'+c]]

                clean_df['remove'][clean_df['dif_'+c] > (2 * clean_df['sd_'+c])] = True 
            clean_df = clean_df[~clean_df['remove']]
        
        df_x = pd.DataFrame()
        df_y = pd.DataFrame()
        df_x[on] = clean_df[on].values
        df_y[on] = clean_df[on].values
        df_x[col_name] = clean_df[col_name + '_x'].values
        df_y[col_name] = clean_df[col_name + '_y'].values        
            
       
        return df_x, df_y

    def get_sample(self, start, end):
        assert start < end <= len(self.x), 'Error:Invalid Indexing.'
        x_sample    = self.x[start:end]
        y_sample    = self.y[start:end]
        time_sample = self.timestamp[start:end]
        return x_sample, y_sample, time_sample

        
    def backtest(self, start, end, entryZ, exitZ, transaction_cost, lag, resample, rounding,minHalflife):
        
        #############################################################
        # INPUT:
        # s1: the symbol of contract one
        # s2: the symbol of contract two
        # x: the price series of contract one
        # y: the price series of contract two
        # OUTPUT:
        # Trade records
        # run regression to find hedge ratio and then create spread series
        
        x, y, time = self.get_sample(start, end)
        d = pd.DataFrame({'y':y,'x':x})
        d.index = pd.to_datetime(time)
        
        d['hr'] = - self.state_means[:,0]
      
        d = d.resample(str(resample)+'T', label='right', convention = 's').mean().dropna()
        
        #Round to chosen number of decimals
        d['hr'] = d['hr'].round(rounding)
        d['spread'] = d.y + (d.x * d.hr)
        
        
        # calculate half life
        halflife = self.half_life(d['spread'],minHalflife)
        
        # calculate z-score with window = half life period
        meanSpread = d.spread.rolling(window=halflife).mean()
        stdSpread = d.spread.rolling(window=halflife).std()
        d['zScore'] = ((d.spread-meanSpread)/stdSpread).shift(lag)

        ##############################################################
        # trading logic
        entryZscore = entryZ
        exitZscore = exitZ

        #set up num units long
        d['long entry'] = ((d.zScore < - entryZscore) & ( d.zScore.shift(1) > - entryZscore))
        d['long exit'] = ((d.zScore > - exitZscore) & (d.zScore.shift(1) < - exitZscore)) 
        d['num units long'] = np.nan 
        d.loc[d['long entry'],'num units long'] = 1 
        d.loc[d['long exit'],'num units long'] = 0 

        d['num units long'][0] = 0 

        d['num units long'] = d['num units long'].fillna(method='pad') #set up num units short 
        d['short entry'] = ((d.zScore > entryZscore) & ( d.zScore.shift(1) < entryZscore))
        d['short exit'] = ((d.zScore < exitZscore) & (d.zScore.shift(1) > exitZscore))
        d.loc[d['short entry'],'num units short'] = -1
        d.loc[d['short exit'],'num units short'] = 0
        d['num units short'][0] = 0
        d['num units short'] = d['num units short'].fillna(method='pad')

        d['numUnits'] = d['num units long'] + d['num units short']

        # Boolean whether transaction occurred
        d['transaction'] = d.numUnits.shift(1) != d.numUnits

        # Take the exception of a switch in the same minute
        d['positionSwitch'] = (d.numUnits.shift(1) == (-d.numUnits)) & d['transaction']

        # Cost of transaction
        d['tradecosts'] = (d['transaction'] *1 + d['positionSwitch']*1 ) * transaction_cost

        # Save hr during holding period
        d['hr_memory'] = np.nan
        buy_sell_rows = d['transaction'] & d['numUnits'] != 0
        d['hr_memory'][buy_sell_rows] = d.hr[buy_sell_rows]
        d['hr_memory'].fillna(method='ffill',inplace=True)

        # Save investment amount during holding period
        d['invest_memory'] = np.nan
        d['invest_memory'][d['transaction'] & d['numUnits'] != 0] = \
                                ((d['x'] * abs(d['hr'])) + d['y'])[d['transaction'] & d['numUnits'] != 0]
        d['invest_memory'].fillna(method='ffill',inplace=True)

        # Save spread according to hr_memory
        d['spreadmemory'] = d.y + (d.x * d.hr_memory)

        # Calculate spread percent change
        d['spread pct ch'] = (d['spreadmemory'] - d['spreadmemory'].shift(1)) / d['invest_memory']
        d['port rets'] = d['spread pct ch'] * d['numUnits'].shift(1) - (d['tradecosts'] /d['invest_memory'])

        # Account for the position switch
        d['port rets'][d['positionSwitch']] = (((d.y + (d.x * d.hr_memory.shift(1))\
                                    - d['spreadmemory'].shift(1)) / d['invest_memory'].shift(1))\
                                    * d['numUnits'].shift(1) - (d['tradecosts'] /d['invest_memory'].shift(1)))\
                                    [d['positionSwitch']]

        # Calculate returns
        d['cum rets'] = d['port rets'].cumsum()
        d['cum rets'] = d['cum rets'] + 1
        d['cum rets'] = d['cum rets'].fillna(1)
        d['port rets'] = d['port rets'].fillna(0)
        
        return d, halflife

    @staticmethod
    def plot_signals(d, fromDate, toDate, ratio):
        #idx = pd.date_range(fromDate,toDate, freq="1min")

        d = d[fromDate:toDate]
        #d = d.reindex(idx, fill_value= np.nan)
        d.index = d.index.map(str)
        # Plot the prices and buy and sell signals from z score

        S = d.y - d.x * ratio

        S.plot(color='b')

        buyS =  np.nan*S.copy()
        sellS = np.nan*S.copy()
        exitL = np.nan*S.copy()
        exitS = np.nan*S.copy()

        longentry = d['long entry'] * d.transaction * (d.numUnits == 1)
        longexit = d['long exit'] * d.transaction * (d.numUnits.shift(1) == 1)
        shortentry = d['short entry'] * d.transaction * (d.numUnits == -1)
        shortexit = d['short exit'] * d.transaction * (d.numUnits.shift(1) == -1)

        buyS[longentry] = S[longentry]
        sellS[shortentry] = S[shortentry]
        exitL[longexit] = S[longexit]
        exitS[shortexit] = S[shortexit]

        buyS.plot(color='g', linestyle='None', marker='o')
        sellS.plot(color='r', linestyle='None', marker='o')
        exitL.plot(color='g', linestyle='None', marker='x')
        exitS.plot(color='r', linestyle='None', marker='x')

        x1,x2,y1,y2 = plt.axis()

        plt.axis((x1,x2,S.min(),S.max()))

        plt.legend(['LOP Spread', 'Enter Long', 'Enter Short','Exit Long', 'Exit Short'])
        plt.xticks(rotation=45, ha="right")

        plt.show()

        print('{} percent return in time window'.format(round(d['port rets'].sum() *100,2)))

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
        return state_means

    @staticmethod
    def half_life(spread,minHalflife):
        spread_lag = spread.shift(1)
        spread_lag.iloc[0] = spread_lag.iloc[1]
        spread_ret = spread - spread_lag
        spread_ret.iloc[0] = spread_ret.iloc[1]
        spread_lag2 = sm.add_constant(spread_lag)
        model = sm.OLS(spread_ret,spread_lag2)
        res = model.fit()
        halflife = int(round(-np.log(2) / res.params[1],0))

        if halflife <= minHalflife:
            halflife = minHalflife
            
        return halflife

    @staticmethod
    def get_indices(index, size):
        #assert n_hist <= index, 'Error:Invalid number of historical observations.'
        start_hist    =  0
        end_hist      =  size#index
        return start_hist, end_hist

    def calibrate(self, start, end):
        x, y, time = self.get_sample(start, end)
        
        a = self.KalmanFilterAverage(x,time)
        b = self.KalmanFilterAverage(y,time)
        self.state_means = self.KalmanFilterRegression(a,b)
        
    def process(self, transaction_cost, entryZ, exitZ, rounding, lag = 0, \
                resample = 1, minHalflife = 1, index=None, **kwargs):
        index = random.randint(0, len(self.x)) if index is None else index
        start_hist, end_hist = self.get_indices(index, len(self.x))
        self.record, self.halflife = self.backtest(start_hist, end_hist, entryZ, exitZ, \
                                           transaction_cost, lag, resample,rounding,minHalflife)
        self.reward = sum(self.record['port rets'])
        
