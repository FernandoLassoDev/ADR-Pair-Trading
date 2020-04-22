import pandas as pd
import numpy as np
from STRATEGY.BaseTradeEngine import BaseTradeEngine
# Visualizations
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import copy

# Search through a range of settings by optimizing over a pair of two parameters at a time
def optimize_parameters_heuristic(engine, settings, values, order, description = None,
                    stat = 'daily_sharpe', maximize = True, train_rng = [0,0.6]):
    # Save figure outputs
    figs = []
    
    # Optimize in given order of pairs
    for o in order:
        v1 = o[0]
        v2 = o[1]
        
        # Plot the sharpe ratio for the given value combinations
        v1_opt, v2_opt, _, fig = engine.plot_param_search(
            settings, v1, values[v1], v2, values[v2], stat, maximize, train_rng)
        
        figs = figs + [fig]
        
        settings.update({v1: v1_opt, v2: v2_opt})
    
    if description is not None:
        pp = PdfPages('OUTPUT/Graphs/'+description+'.pdf')
        for fig in figs:
            fig.show()
            pp.savefig(fig, bbox_inches="tight")
        pp.close()
        
    return settings

# Backtest two stocks as a pair and 
def backtest_model(odr, adr, engineClass, settings, treatment,  values, order, version = 'model',
                   clean_version = 'treatment', bid = 'bid', ask = 'ask', train = [0,0.75], test = [0.75,1]):

    
    print('------------------------------------------')
    print(odr + ' vs ' + adr + ' pair trading using ' + version)
    print('------------------------------------------')
    
    # Read prices
    x = pd.read_csv('STATICS/PRICE/'+odr+'.csv', index_col = 'date' )
    y = pd.read_csv('STATICS/PRICE/'+adr+'.csv', index_col = 'date' )

    print(('Observations in source  | '+adr+' : {a} , '+odr+' : {o} ').format(a = len(x), o = len(y)))

    a, b = BaseTradeEngine.clean_data(x, y, 'last', rm_outliers = False, rm_wide_spread = False)
    print(('After merging on date   | '+adr+' : {a} , '+odr+' : {o} ').format(a = len(a), o = len(b)))
    
    treatment.pop('is_cleaned', 0)
    x, y = BaseTradeEngine.clean_data(x, y, 'last', **treatment)
    print(('After applying filters  | '+adr+' : {a} , '+odr+' : {o} ').format(a = len(a), o = len(b)))
     
    engine  = engineClass(x, y, 'last', bid, ask, is_cleaned = True, **settings)

    d = odr + '_' + adr + '_' + version + '_' + clean_version 
    settings_opt = optimize_parameters_heuristic(engine, copy.deepcopy(settings), values, order, description = d)
    
    print(settings_opt)

    # In sample results
    engine.process(train_rng = train,**settings_opt)
    #engine.plot_signals("2018-09-01","2018-10-01")
    train_results = engine.get_summary(verbose = False)
    train_results.index = [(odr + '_' + adr,'train',version,clean_version)]
    
    # Out of sample results
    engine.process(train_rng = test,**settings_opt)
    #engine.plot_signals("2018-09-01","2018-10-01")
    test_results = engine.get_summary(verbose = False)
    test_results.index = [(odr + '_' + adr,'test',version,clean_version)]
    
    return train_results.transpose(), test_results.transpose() , settings_opt