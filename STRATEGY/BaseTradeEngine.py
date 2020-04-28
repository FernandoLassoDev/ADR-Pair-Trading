import pandas as pd
import numpy as np
import ffn
from copy import deepcopy

# Visiualization tools
from tqdm import tqdm
from itertools import product
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from tabulate import tabulate

class BaseTradeEngine(object):
    
    def __init__(self, *args, **kwargs):
        
        x = args[0]
        y = args[1]
        self.col_name = args[2]
        self.bid = args[3]
        self.ask = args[4]
        
        # Defaults in case they weren't given
        kwargs.setdefault('is_cleaned', False)
        kwargs.setdefault('rm_outliers', True)
        kwargs.setdefault('rm_wide_spread', True)
        kwargs.setdefault('max_width', 20)
        kwargs.setdefault('adjust_spread', True)
        kwargs.setdefault('adjust_width', 10)
        kwargs.setdefault('transaction_cost', 0.0063)
        kwargs.setdefault('lag', 0)
        kwargs.setdefault('resample', 1)
        kwargs.setdefault('rounding', 3)
        
        self.resample = int(kwargs['resample'])
        self.transaction_cost = kwargs['transaction_cost']
        
        if kwargs['is_cleaned'] is not True:
            x, y = self.clean_data(x, y, self.col_name, self.bid, self.ask,
                                   kwargs['rm_outliers'],kwargs['rm_wide_spread'],
                                   kwargs['max_width'], kwargs['adjust_spread'],kwargs['adjust_width'])
        
        # Save for future resampling
        self.original_x = x
        self.original_y = y
        
        self.resampling(self.resample)
        
        # Calibrate state means
        self.calibrate(**kwargs)  
            
        self._reward = 0
        self._record = None
    
        
    def process(self):
        pass
    
    def calibrate(self):
        pass
    
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
    
    ############################
    ### BACKTESTING ENGINE    ##
    ############################

    # A fully vectorized backtest engine for pair trading with minute level bid ask data
    def backtest(self, buy_x, buy_y, sell_x, sell_y, time, **kwargs):

        #############################################################
        # Iniate the trade record with prices and signals to react on
        
        # Trade record will be constructed from here
        d = pd.DataFrame({'buy_x':buy_x,'buy_y':buy_y,'sell_x':sell_x,'sell_y':sell_y})
        
        # Set the time index
        d.index = pd.to_datetime(time)
        
        # Give the hedge ratio
        d = self.calculate_hr(d, **kwargs)

        # Calculate statistic for buy and selling spread
        d['buy_spread']      = d.buy_y  - np.multiply(d.sell_x, d.hr)
        d['sell_spread']     = d.sell_y - np.multiply(d.buy_x,  d.hr)
        
        # Calculate the theoretical investment needed for buying or selling a spread
        d['buy_investment']  = d.buy_y  + np.multiply(d.sell_x, d.hr)
        d['sell_investment'] = d.sell_y + np.multiply(d.buy_x,  d.hr)
        
        # Bid ask spread width in bps
        d['bid_ask_bps']  = 10000 * (d.buy_spread - d.sell_spread) / d.buy_investment
        
        d    = self.calculate_signals(d, **kwargs)
        d    = self.calculate_thresholds(d, **kwargs)
        
        ##############################################################
        # trading logic
        
        # Enter and exit decisions
        d = self.calculate_entry_exit(d)
        
        # Don't cross spread if it is too large
        d.long_entry[d['bid_ask_bps'] > 15] = False
        d.short_entry[d['bid_ask_bps'] > 15] = False
        
        # Calculate units long at every time
        d['num_units_long'] = np.nan 
        d.num_units_long.loc[d.long_entry] = 1 
        d.num_units_long.loc[d.long_exit]  = 0
        d.num_units_long[0] = 0 
        d.num_units_long = d.num_units_long.fillna(method='pad')

        # Calculate units short at every time 
        d['num_units_short'] = np.nan 
        d.num_units_short.loc[d.short_entry] = -1
        d.num_units_short.loc[d.short_exit]  = 0
        d.num_units_short[0] = 0 
        d.num_units_short = d.num_units_short.fillna(method='pad')

        # Calculate holdings at every time
        d['numUnits'] = d.num_units_long + d.num_units_short

        # Boolean whether transaction occurred
        d['transaction'] = d.numUnits.shift(1) != d.numUnits
        d['transaction'].iloc[0] = False
        
        # Account for the exception of a switch in the same minute
        d['positionSwitch'] = (d.numUnits.shift(1) == (-d.numUnits)) & d.transaction

        # Cost of transaction
        d['tradecosts'] = (d.transaction *1) * kwargs['transaction_cost']

        # Save hr during each holding period
        d['hr_memory'] = np.nan
        buy_rows  = (d.numUnits > 0) & d.transaction
        sell_rows = (d.numUnits < 0) & d.transaction
        d.hr_memory[buy_rows | sell_rows] = d.hr[buy_rows | sell_rows]
        d.hr_memory.fillna(method='ffill',inplace=True)
        d.hr_memory[d.hr_memory.isnull()] = d.hr[d.hr_memory.isnull()] #This doesnt matter

        # Save investment amount during holding period
        d['invest_memory'] = np.nan
        d.invest_memory[buy_rows]  = d.buy_investment[buy_rows]
        d.invest_memory[sell_rows] = d.sell_investment[sell_rows]
        d.invest_memory.fillna(method='ffill',inplace=True)
        d.invest_memory[d.invest_memory.isnull()] = d.buy_investment[d.invest_memory.isnull()] #This doesnt matter

        # Save spread according to hr_memory
        d['buy_spread_memory']  = d.buy_y  - (d.sell_x * d.hr_memory)
        d['sell_spread_memory'] = d.sell_y - (d.buy_x  * d.hr_memory)
        d['buy_spread_memory_pos_switch']  = d.buy_y  - (d.sell_x * d.hr_memory.shift(1))
        d['sell_spread_memory_pos_switch'] = d.sell_y - (d.buy_x  * d.hr_memory.shift(1))

        # Calculate spread percent change
        d['buy_spread_pct_ch']  = (d.buy_spread_memory  - d.buy_spread_memory.shift(1))  / d.invest_memory
        d['sell_spread_pct_ch'] = (d.sell_spread_memory - d.sell_spread_memory.shift(1)) / d.invest_memory

        # Account for position switch return
        d['buy_spread_pct_ch'][d.positionSwitch] = ((d.buy_spread_memory_pos_switch  - 
                                                     d.sell_spread_memory.shift(1)) / 
                                                    d.invest_memory.shift(1))[d.positionSwitch]
        d['sell_spread_pct_ch'][d.positionSwitch] = ((d.sell_spread_memory_pos_switch  - 
                                                      d.sell_spread_memory.shift(1)) / 
                                                    d.invest_memory.shift(1))[d.positionSwitch]
        
        # Return for long position is change in the selling spread price and for short vice versa
        d['long_holding_ret'] = 0
        d['short_holding_ret'] = 0
        d.long_holding_ret[d.num_units_long.shift(1) > 0]  =   d.sell_spread_pct_ch[d.num_units_long.shift(1) > 0]
        d.short_holding_ret[d.num_units_short.shift(1) < 0] = - d.buy_spread_pct_ch[d.num_units_short.shift(1) < 0]
        
        # When buying or selling, immediately lose current spread
        d['spread_cost_ret'] = 0
        d.spread_cost_ret[buy_rows | sell_rows] =  ((d.buy_spread_memory - d.sell_spread_memory) / 
                                                   d.invest_memory)[buy_rows | sell_rows] # divided by investment
       
        # Compute relative transaction cost
        d['transaction_cost_ret'] = d.tradecosts / d.invest_memory
        # Treat the exceptional case of a position switch: account for double transaction cost
        d.transaction_cost_ret[d.positionSwitch] = (d.transaction_cost_ret + 
                                                    d.tradecosts[d.positionSwitch] / 
                                          d.invest_memory.shift(1))[d.positionSwitch]
        
        # Calculate portfolio returns
        d['port_rets'] = (1 + d.long_holding_ret) * (1 + d.short_holding_ret) * \
                         (1 - d.spread_cost_ret ) * (1 - d.transaction_cost_ret)
      
        d['cum rets'] = d.port_rets.cumprod()
        
        return d

    def calculate_entry_exit(self, d):
        d['long_entry'] =  ((d.buySignal > d.entryThreshold ) & ( d.buySignal.shift(1)  < d.entryThreshold)) 
        d['long_exit']  = (((d.buySignal < d.exitThreshold)  &  ( d.buySignal.shift(1)  > d.exitThreshold)) | #Stoploss
                       ((d.sellSignal > d.entryThreshold) & ( d.sellSignal.shift(1) < d.entryThreshold))) #Positionswitch
        d['long_exit'][-1] = True
        # Set up num units short
        d['short_entry'] =  ((d.sellSignal > d.entryThreshold) & ( d.sellSignal.shift(1) < d.entryThreshold))
        d['short_exit']  = (((d.sellSignal < d.exitThreshold)  & ( d.sellSignal.shift(1) > d.exitThreshold)) | #Stoploss
                        ((d.buySignal  > d.entryThreshold) & ( d.buySignal.shift(1)  < d.entryThreshold))) #Positionswitch
        d['short_exit'][-1] = True
        
        return d
        
        
    ############################
    ### Cleaning              ##
    ############################
    
    # A method to clean data by merging the two data sets on time and potentially removing outliers 
    @classmethod
    def clean_data(cls, x, y, col_name, bid = "bid", ask = "ask",  rm_outliers = True, 
                   rm_wide_spread = True, max_width = 15, adjust_spread = False, adjust_width = 10):
       
        x.replace([np.inf, -np.inf], np.nan, inplace=True)
        y.replace([np.inf, -np.inf], np.nan, inplace=True)
       
        clean_df = pd.merge(left=x, right=y,  how='inner', left_index = True, right_index = True)
        
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
            del(clean_df['remove'])
        
        # Modify spreads
        clean_df['remove'] = False
        for c in ['x','y']:
            clean_df['spread_'+c] = clean_df[ask + '_'+c] - clean_df[bid + '_'+c]  
            clean_df['bps_spread_'+c] = 10000 * clean_df['spread_'+c] / clean_df[col_name + '_'+c]

            if rm_wide_spread:
                #Delete spreads above max_width
                clean_df['remove'][clean_df['bps_spread_'+c] > max_width] = True 
                
            if adjust_spread:
                #Create new bid ask width given allowed width
                clean_df['new_'+ask + '_'+c] = clean_df[col_name + '_'+c] * (1+adjust_width/20000)
                clean_df['new_'+bid + '_'+c] = clean_df[col_name + '_'+c] * (1-adjust_width/20000)
                up_down_adj = np.max(clean_df['new_'+ask +'_'+c]-clean_df[ask +'_'+c],0)+\
                              np.min(clean_df['new_'+bid +'_'+c]-clean_df[bid +'_'+c],0)
                clean_df['new_'+ask + '_'+c] = clean_df['new_'+ask + '_'+c] - up_down_adj
                clean_df['new_'+bid + '_'+c] = clean_df['new_'+bid + '_'+c] - up_down_adj
                chng = clean_df['bps_spread_'+c] > adjust_width
                clean_df[ask +'_'+c][chng] = clean_df['new_'+ask + '_'+c][chng] 
                clean_df[bid +'_'+c][chng] = clean_df['new_'+bid + '_'+c][chng] 
                
        clean_df = clean_df[~clean_df['remove']]
        del(clean_df['remove'])
            
          
        # Resample if necessary    
        clean_df.index = pd.to_datetime(clean_df.index)
        
        df_x = clean_df.iloc[:, list(range(0,x.shape[1])) ]
        df_y = clean_df.iloc[:, list(range(x.shape[1],2*x.shape[1])) ]
        
        df_x.columns = [col.replace('_x','') for col in df_x.columns]
        df_y.columns = [col.replace('_y','') for col in df_y.columns]
       
        return df_x, df_y
    
    ############################
    ### DATA SUBSAMPLING      ##
    ############################
    
    # Method to resample data
    def resampling(self, resample):
        
        x = self.original_x.resample(str(resample)+'T', label='right', convention = 's').mean().dropna()
        y = self.original_y.resample(str(resample)+'T', label='right', convention = 's').mean().dropna()
        
        self.timestamp = x.index
        self.buy_x = x[self.ask].values.reshape(-1, )
        self.buy_y = y[self.ask].values.reshape(-1, )
        self.sell_x = x[self.bid].values.reshape(-1, )
        self.sell_y = y[self.bid].values.reshape(-1, )
        self.x = x[self.col_name].values.reshape(-1, )
        self.y = y[self.col_name].values.reshape(-1, )
        
     # Method to retrieve subsample
    def get_sample(self, start, end, *args):
        assert start < end <= len(self.x), 'Error:Invalid Indexing.'
        samples = []
        for arg in args:
            samples.append(arg[start:end])
        return samples
    
    # Method to retrieve indices of start and end of data to predict
    def get_indices(self, train_range):
        #assert n_hist <= index, 'Error:Invalid number of historical observations.'
        start_hist    =  int(len(self.x) * train_range[0])
        end_hist      =  int(len(self.x) * train_range[1]) -1
        
        return start_hist, end_hist

    ############################
    ### PERFORMANCE RECORD    ##
    ############################
    
    def get_summary(self, fromDate = None, toDate = None, trim = False, verbose = True):
        
        if trim:
            d = self.record[fromDate:toDate]
        else:
            d = self.record
            
        longentry  = d.long_entry  & d.transaction & (d.numUnits == 1)
        longexit   = d.long_exit   & d.transaction & (d.numUnits.shift(1) == 1)
        shortentry = d.short_entry & d.transaction & (d.numUnits == -1)
        shortexit  = d.short_exit  & d.transaction & (d.numUnits.shift(1) == -1)
        
        if sum(longentry)  > sum(longexit) :   longexit[-1] = True
        if sum(shortentry) > sum(shortexit):  shortexit[-1] = True
        if sum(longentry)  < sum(longexit) :   longentry[0] = True
        if sum(shortentry) < sum(shortexit):  shortentry[0] = True
        
        if(sum(longentry) + sum(shortentry)) == 0:
            print("No transactions made!")
            return pd.DataFrame.from_records([{ 'sharpe_ratio'  : 0,'total_return'  : 0,
                   'max_drawdown'  : 0,'long_ret'      : 0, 'short_ret'     : 0,'spread_ret'    : 0,
                   'trans_ret'     : 0,'n_transactions': 0,'mean_hold_time': 0}])
       
        l = {'entry': d.index[longentry], 'exit': d.index[longexit]}
        s = {'entry': d.index[shortentry], 'exit': d.index[shortexit]}
        
        long_holdings  = pd.DataFrame(l)
        short_holdings = pd.DataFrame(s)
        long_holdings['position'] = 'long'
        short_holdings['position'] = 'short'
        holdings = pd.concat([long_holdings, short_holdings])

        holdings['time'] = holdings['exit'] - holdings['entry']
        
        perf = ffn.PerformanceStats(d['cum rets'], rf = 0.0016)
        
        timing = np.mean(holdings['time']).components
        minutes_hold = timing.days * 24 * 60 + timing.hours *60 + timing.minutes
           
        out = pd.DataFrame.from_records([{
                   'sharpe_ratio'  : round(perf.stats['daily_sharpe']        , 3),
                   'total_return'  : round(perf.stats['total_return']        , 3),
                   'max_drawdown'  : round(perf.stats['max_drawdown']        , 3),
                   'long_ret'      : round((1+d.long_holding_ret).prod()     , 3),
                   'short_ret'     : round((1+d.short_holding_ret).prod()    , 3),
                   'spread_ret'    : round((1-d.spread_cost_ret).prod()      , 3),
                   'trans_ret'     : round((1-d.transaction_cost_ret).prod() , 3),
                   'n_transactions': len(holdings),
                   'mean_hold_time'     : minutes_hold
                      }])
        
        if verbose:
            print(tabulate(out[
                ['sharpe_ratio','long_ret','short_ret','spread_ret','trans_ret',
                 'n_transactions','mean_hold_time']], headers='keys', tablefmt='psql'))
            perf.display()
            perf.plot()           
        
        return out
    
    
    ############################
    ### DATA VISUALIZATION    ##
    ############################

    # Visualize the trading decision
    def plot_signals(self, fromDate, toDate, ratio = None):
            
        d = self.record[fromDate:toDate]
        #d.index = d.index.map(str)
        
        if ratio is None:
            ratio = d.hr
            holding = (d.numUnits != 0) | (d.numUnits.shift(1) != 0)
            ratio[holding] =d.hr_memory[holding]

        # Plot buy spread price
        S_buy  = d.buy_y - d.sell_x * ratio

         # Plot sell spread price
        S_sell = d.sell_y - d.buy_x * ratio
        
        buyS =  np.nan*S_buy.copy()
        exitS = np.nan*S_buy.copy()
        sellS = np.nan*S_sell.copy()
        exitL = np.nan*S_sell.copy()
        
        # Plot long positions enter and sell positions exit
        longentry = d.long_entry & d.transaction & (d.numUnits == 1)
        shortexit = d.short_exit & d.transaction & (d.numUnits.shift(1) == -1)
   
        # Plot short positions enter and long positions exit
        shortentry = d.short_entry & d.transaction & (d.numUnits == -1)
        longexit   = d.long_exit   & d.transaction & (d.numUnits.shift(1) == 1)

        #Select
        buyS[longentry] = S_buy[longentry]
        exitS[shortexit] = S_buy[shortexit]
        sellS[shortentry] = S_sell[shortentry]
        exitL[longexit]   = S_sell[longexit]
        
        S_buy.plot(color='k')
        S_sell.plot(color='b')
        
        buyS.plot(color='g' , linestyle='None', marker='o')
        sellS.plot(color='r', linestyle='None', marker='o')
        exitL.plot(color='g', linestyle='None', marker='x')
        exitS.plot(color='r', linestyle='None', marker='x')

        # Make graph
        x1,x2,y1,y2 = plt.axis()
       
        plt.axis((x1,x2,S_sell.min(),S_buy.max()))

        plt.legend(['Buy Spread','Sell Spread', 'Enter Long', 'Enter Short', 'Exit Long', 'Exit Short'])
        plt.xticks(rotation=45, ha="right")

        plt.show()

        print('{} percent return in time window'.format(round((d.port_rets.prod()-1) *100,2)))

    # Calculate the benchmark performance for a given range of parameter values
    def calculate_benchmark(self, default, v1_name, v1_values, v2_name = None, v2_values = None, train_rng = [0,0.8]):

        #Copy dictionary
        param_dict = {}
        for key in default:
            param_dict[key] = [default[key]]
        
        #Set the parameter ranges
        param_dict[v1_name] = v1_values
        if(v2_values is not None):
            param_dict[v2_name] = v2_values
        
        #Create output dataframe
        cols = list(param_dict.keys())
        cols.extend(['perf'])
        benchmark_output = pd.DataFrame(columns = cols)
        
        #Process for the value combinations
        for p in tqdm(list(product(*param_dict.values()))):
            
            a = {}
            for i,key in enumerate(param_dict.keys()):
                a[key] = p[i]
            
            self.process(train_rng = deepcopy(train_rng), **a)
            trade_record = self.record
            
            a['perf'] = ffn.PerformanceStats(trade_record['cum rets'], rf = 0.0016)
            benchmark_output = benchmark_output.append(a, ignore_index=True)

        return benchmark_output

    def trisurf_plot(self,x,y,z,v1,v2,stat,txt):
        
        fig = plt.figure()

        ax = fig.gca(projection='3d')

        surf = ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0, \
                               antialiased=True, label=stat+' statistic for ' + v1 + ' vs ' + v2)
        fig.colorbar(surf)

        ax.set_xlabel(v1, fontsize=15)
        ax.set_ylabel(v2, fontsize=15)
        ax.set_zlabel(stat, fontsize=15)
        
        fig.text(.5, -0.05, txt, ha='center')
        
        fig.tight_layout()
        plt.show()
        
        return fig

    # Plot a statistic like the sharpe ratio of the benchmark for a given range of parameter values
    def plot_param_search(self, default_dict, v1_name, v1_values, v2_name = None, v2_values = None, \
                    stat = 'daily_sharpe', maximize = True, train_rng = [0,0.8]):

        is_3d = v2_name is not None

        out_df = self.calculate_benchmark(default_dict, v1_name, v1_values, v2_name, v2_values, train_rng)
        
        n =  len(v1_values) *  len(v2_values)
        x = np.ones(n)
        y = np.ones(n)
        z = np.ones(n)
        count = 0

        for i in v1_values:
            for j in v2_values:
                default_dict[v1_name] = i
                default_dict[v2_name] = j

                x[count] = i
                y[count] = j
                
                perf = out_df.loc[(out_df[list(default_dict)] == pd.Series(default_dict)).all(axis=1)]['perf'].iloc[0]
                z[count] = perf.stats[stat]
                if z[count] == "nan" or z[count] =="inf" or z[count] =="-inf" : z[count] = 0
                count = count + 1

        maxIdx = z.argmax() if maximize else z.argmin()
        
        v1_opt_value = round(x[maxIdx],2)
        v2_opt_value = round(y[maxIdx],2)
        stat_value = round(z[maxIdx],3)
        
        caption = 'Best ' + stat + ' reached at {s}'.format(s = stat_value)+ \
              ' for ' + v1_name + ' and ' + v2_name + \
              ' equal to {a} and {b}'.format(a = v1_opt_value, b = v2_opt_value)
        
        fig = self.trisurf_plot(x,y,z,v1_name,v2_name,stat,caption)
   
        return v1_opt_value, v2_opt_value, stat_value, fig
