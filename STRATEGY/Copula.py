import pandas as pd
import numpy as np
import sys
import random
from STRATEGY.BaseTradeEngine import BaseTradeEngine
from statsmodels.distributions.empirical_distribution import ECDF
from statsmodels.regression.rolling import RollingOLS
from scipy import stats
from scipy.optimize import minimize
from scipy import integrate as ig

def get_src_cls(source_name):
    return getattr(sys.modules[__name__], source_name)

class Copula(BaseTradeEngine):
    
    def __init__(self, *args, **kwargs):
        super(Copula, self).__init__(*args, **kwargs)
        
    # Run a backtest with given parameter inputs
    def process(self, windowOLS = 150,  copula_lookback = 150, recalibrate_n = 50, 
                 cap_CL = 0.95, floor_CL = 0.05, lag = 0, rounding = 3, resample = 1, train_rng = [0,1], **kwargs):
       
        minWindowOLS =  int(min(max(windowOLS/np.sqrt(resample),10),len(self.original_x)/2))
        
        minCopula    =  int(min(max(copula_lookback/np.sqrt(resample),recalibrate_n),len(self.original_x)/2))
        minRecalib   =  int(max(recalibrate_n/np.sqrt(resample),5))
            
        # Calibrate OLS if necessary
        if minCopula != self.copula_lookback or minRecalib != self.recalibrate_n  \
                 or minWindowOLS != self.windowOLS or resample != self.resample:
            self.resampling(resample)
            self.calibrate(minWindowOLS,minCopula,minRecalib)
            self.resample = resample
        
        # Get start and end time for data
        start_hist, end_hist = self.get_indices(train_rng)
        
        #In any case, start data after enough data gathered for the OLS window
        min_start = (max(minWindowOLS,minCopula)-1) / len(self.x) 
        train_rng[0] = max(train_rng[0],min_start)
        
        subsamples = [self.buy_x , self.buy_y, self.sell_x, self.sell_y, self.beta, \
                      self.timestamp, self.MI_u_v, self.MI_v_u]
        
        [buy_x, buy_y, sell_x, sell_y, beta, time, MI_u_v, MI_v_u]  = \
                            self.get_sample(start_hist, end_hist, *subsamples)
        
        parameters = {'beta': beta, 'floor_CL':floor_CL , 'cap_CL': cap_CL,
                      'lag': lag, 'rounding': rounding, 'MI_u_v': MI_u_v, 'MI_v_u': MI_v_u}
        
        parameters.update(kwargs)
        
        # Create the record
        self.record = self.backtest(buy_x, buy_y, sell_x, sell_y, time, **parameters)
        
        # Legacy
        self.reward = sum(self.record.port_rets)
        
       
    def calibrate(self, windowOLS, copula_lookback, recalibrate_n, **kwargs):
        self.windowOLS = int(windowOLS)
        self.copula_lookback = int(copula_lookback)
        self.recalibrate_n = int(recalibrate_n)
        
        df = pd.DataFrame({'y':self.y,'x':self.x,'c':1})
        
        model = RollingOLS(endog =df['y'], exog=df['x'],window=self.windowOLS)
        rres = model.fit()
      
        self.beta = rres.params['x'].values.reshape(-1, )
        
        # Copula decision:
        df['x_log_ret']= np.log(df.x) - np.log(df.x.shift(1))
        df['y_log_ret']= np.log(df.y) - np.log(df.y.shift(1))
        
        # Convert the two returns series to two uniform values u and v using the empirical distribution functions
        ecdf_x, ecdf_y  = ECDF(df.x_log_ret), ECDF(df.y_log_ret)
        u, v = [ecdf_x(a) for a in df.x_log_ret], [ecdf_y(a) for a in df.y_log_ret]
        
        # Compute the Akaike Information Criterion (AIC) for different copulas and choose copula with minimum AIC
        tau = stats.kendalltau(df.x_log_ret, df.y_log_ret)[0]  # estimate Kendall'rank correlation
        AIC ={}  # generate a dict with key being the copula family, value = [theta, AIC]

        for i in ['clayton', 'frank', 'gumbel']:
            param = self._parameter(i, tau)
            lpdf = [self._lpdf_copula(i, param, x, y) for (x, y) in zip(u, v)]
            # Replace nan with zero and inf with finite numbers in lpdf list
            lpdf = np.nan_to_num(lpdf) 
            loglikelihood = sum(lpdf)
            AIC[i] = [param, -2 * loglikelihood + 2]
        # Choose the copula with the minimum AIC
        copula = min(AIC.items(), key = lambda x: x[1][1])[0]
        
        self.startIdx = copula_lookback + 1 # Because first is NAN
        
        df['MI_u_v'] = 0.5
        df['MI_v_u'] = 0.5
        
        for i in np.arange(self.startIdx , len(df)-recalibrate_n, recalibrate_n):
            
            window = range(i - copula_lookback, i) 
            predWindow = range(i, i + recalibrate_n)
            
            x_hist = df.x_log_ret.iloc[window]
            y_hist = df.y_log_ret.iloc[window]
            x_forw = df.x_log_ret.iloc[predWindow]
            y_forw = df.y_log_ret.iloc[predWindow]
            
            # Estimate Kendall'rank correlation
            tau = stats.kendalltau(x_hist, y_hist)[0] 

            # Estimate the copula parameter: theta
            theta = self._parameter(copula, tau)

            # Simulate the empirical distribution function for returns of selected trading pair
            ecdf_x,  ecdf_y  = ECDF(x_hist), ECDF(y_hist) 

            # Now get future values
            a, b = self._misprice_index(copula, theta, ecdf_x(x_forw), ecdf_y(y_forw))
            
            df.MI_u_v.iloc[predWindow] = a
            df.MI_v_u.iloc[predWindow] = b
                        
        self.MI_u_v = df.MI_u_v
        self.MI_v_u = df.MI_v_u
        
    
    def calculate_hr(self, record, beta, rounding, **kwargs):
        record['hr'] = beta.round(int(rounding))
        return record
        
    def calculate_signals(self, record, MI_u_v, MI_v_u, lag, **kwargs):
        
        record['MI_u_v']  = MI_u_v.values
        record['MI_v_u']  = MI_v_u.values 
          
        return record
    
    
    def calculate_thresholds(self, record, cap_CL, floor_CL, **kwargs):
        record['cap']    = cap_CL
        record['floor']  = floor_CL
        return record
    
    def calculate_entry_exit(self, record):
        
        record['long_entry'] =  (record.MI_u_v > record.cap) & (record.MI_v_u < record.floor)
        record['long_exit']  =  (record.MI_v_u > (record.cap -0.1)) & (record.MI_u_v < (record.floor + 0.1))
        record['long_exit'][-1] = True
        
        # Set up num units short
        record['short_entry'] =  (record.MI_v_u > record.cap) & (record.MI_u_v < record.floor)
        record['short_exit']  =  (record.MI_u_v > (record.cap -0.1)) & (record.MI_v_u < (record.floor+0.1))
        record['short_exit'][-1] = True
        
        #shift n down
        #for i in range(5):
           # record['long_entry'] = record['long_entry'] | record['long_entry'].shift(i)
           # record['short_entry'] = record['short_entry'] | record['short_entry'].shift(i)
            
        return record
    
    def _parameter(self, family, tau):
        ''' Estimate the parameters for three kinds of Archimedean copulas
        according to association between Archimedean copulas and the Kendall rank correlation measure
        '''

        if  family == 'clayton':
            return 2 * tau / (1 - tau)

        elif family == 'frank':

            '''
            debye = quad(integrand, sys.float_info.epsilon, theta)[0]/theta  is first order Debye function
            frank_fun is the squared difference
            Minimize the frank_fun would give the parameter theta for the frank copula 
            ''' 
            integrand = lambda t: t / (np.exp(t) - 1)  # generate the integrand
            frank_fun = lambda theta: ((tau - 1) / 4.0 -(ig.quad(integrand, sys.float_info.epsilon, theta)[0] \
                                                         / theta - 1) / theta) ** 2

            return minimize(frank_fun, 4, method='BFGS', tol=1e-5).x 

        elif family == 'gumbel':
            return 1 / (1 - tau)

    def _lpdf_copula(self, family, theta, u, v):

        if  family == 'clayton':
            pdf = (theta + 1) * ((u ** (-theta) + v ** (-theta) - 1) ** (-2 - 1 / theta)) *\
                                (u ** (-theta - 1) * v ** (-theta - 1))

        elif family == 'frank':
            num = -theta * (np.exp(-theta) - 1) * (np.exp(-theta * (u + v)))
            denom = ((np.exp(-theta * u) - 1) * (np.exp(-theta * v) - 1) + (np.exp(-theta) - 1)) ** 2
            pdf = num / denom

        elif family == 'gumbel':
            A = (-np.log(u)) ** theta + (-np.log(v)) ** theta
            c = np.exp(-A ** (1 / theta))
            pdf = c * (u * v) ** (-1) * (A ** (-2 + 2 / theta)) * ((np.log(u) * np.log(v)) \
                                      ** (theta - 1)) * (1 + (theta - 1) * A ** (-1 / theta))
        return np.log(pdf)

    def _misprice_index(self, family, theta, u, v):

        if  family == 'clayton':
            MI_u_v = v**(-theta-1) * (u**(-theta)+v**(-theta)-1)**(-1/theta-1) # P(U<u|V=v)
            MI_v_u = u**(-theta-1) * (u**(-theta)+v**(-theta)-1)**(-1/theta-1) # P(V<v|U=u)

        elif family == 'frank':
            A = (np.exp(-theta*u)-1) * (np.exp(-theta*v)-1) + (np.exp(-theta*v)-1)
            B = (np.exp(-theta*u)-1) * (np.exp(-theta*v)-1) + (np.exp(-theta*u)-1)
            C = (np.exp(-theta*u)-1) * (np.exp(-theta*v)-1) + (np.exp(-theta)-1)
            MI_u_v = B/C
            MI_v_u = A/C

        elif family == 'gumbel':
            A = (-np.log(u))**theta + (-np.log(v))**theta
            C_uv = np.exp(-A**(1/theta))   # C_uv is gumbel copula function C(u,v)
            MI_u_v = C_uv * (A**((1-theta)/theta)) * (-np.log(v))**(theta-1) * (1/v)
            MI_v_u = C_uv * (A**((1-theta)/theta)) * (-np.log(u))**(theta-1) * (1/u)
            
        return MI_u_v, MI_v_u