'''
Created on 14 de ago de 2019
@author: gusta
'''

# lib to handle with time series
from pandas import Series
# lib to plot the data
import matplotlib.pyplot as plt
# lib to use the decompositio in the time series
from statsmodels.tsa.seasonal import seasonal_decompose
# lib to use the acf and pcf correlations
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# lib to organize the data in data frame
from pandas import DataFrame
# lib to organize the time series by periods
from pandas import TimeGrouper

# to read the time series
series = Series.from_csv('airline-passengers.csv', header=0)
series.plot()
plt.show()

# to plot by year
groups = series.groupby(TimeGrouper('A'))
years = DataFrame()
for name, group in groups:
    years[name.year] = group.values
years.boxplot()
plt.show()

# to plot by month
groups = series.groupby(TimeGrouper('M'))
years = DataFrame()
for name, group in groups:
    years[name.month] = group.values
years.boxplot()
plt.show()

# to decompose the time series
result = seasonal_decompose(series, model='multiplicative')
result.plot()
plt.show()

# to use the acf
plot_acf(series, lags=10)
plt.show()

# to use the pacf
plot_pacf(series, lags=10)
plt.show()

# to plot the acf above the residual time series
#result.trend; result.seasonal; result.resid; result.observed 
plot_acf(result.resid, lags=10)
plt.show()
