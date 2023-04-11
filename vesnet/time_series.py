# TODO: Still under development!!

df_events.columns
df_events['date'] = pd.to_datetime(df_events['start.time'])
dark_df_time_day = df_events['event_id'].groupby(df_events['date'].dt.to_period('D')).count()
dark_df_time_day = pd.DataFrame(dark_df_time_day.resample('D').asfreq().fillna(0))
dark_df_time_day.index=dark_df_time_day.index.to_timestamp()

dark_df_time_month = dark_df_time_day['event_id'].resample('M').mean()
dark_df_time_month = dark_df_time_month.fillna(dark_df_time_month.bfill())
dark_df_time_month

# Time series analysis

# Fit a model to determine any trend, seasonality, cyclicality of the data
from statsmodels.tsa.seasonal import seasonal_decompose

decompose_result_mult = seasonal_decompose(dark_df_time_month)
plt.rcParams.update({'figure.figsize': (10,10)})
decompose_result_mult.plot().suptitle('Additive Decompose', fontsize=22)
plt.show()

import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

warnings.filterwarnings("ignore") # specify to ignore warning messages

for param in pdq:
    for param_seasonal in seasonal_pdq:
            mod = sm.tsa.statespace.SARIMAX(dark_df_time_week.event_id,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            
            
mod = sm.tsa.statespace.SARIMAX(dark_df_time_week.event_id,
                                order=(1, 1, 1),
                                seasonal_order=(0, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()
print(results.summary().tables[1])

results.plot_diagnostics(figsize=(15, 12))
plt.show()

pred = results.predict(start='2023-04-01', stop='2023-04-30')
pred