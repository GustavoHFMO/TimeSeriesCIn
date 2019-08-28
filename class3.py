'''
Created on 28 de ago de 2019
@author: gusta
'''

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AR
import matplotlib.pyplot as plt
from pandas import Series
import numpy as np
import copy

################### simulation of a time series with AR model ##################
# creating a seed
np.random.seed(1)
# mean and standard deviation
mu, sigma, length = 100, 20, 100 
# generating a gaussian white noise
w = np.random.normal(mu, sigma, length)
# generating the lags 
serie = copy.deepcopy(w)
# creating a time series
for t in range(1, length):
    serie[t] = 0.8 * serie[t-1] + w[t]
# plotting the generated time series
plt.plot(serie)
plt.show()
################################################################################

######### analyzing the ACF and PACF on generated time series ##################
# to use the acf
plot_acf(serie, lags=10)
plt.show()

# to use the pacf
plot_pacf(serie, lags=10)
plt.show()
################################################################################

######################## harmonic seasonal models ##############################
# creating a seed
np.random.seed(1)
# mean and standard deviation
mu, sigma, length = 100, 0.5, 120
# generating the time
time = [t for t in range(length)]
# generating a gaussian white noise
w = np.random.normal(mu, sigma, length)
# simulating the trend
trend = [0.1 + 0.005 * t + 0.001 * t**2 for t in time]
# simulating the seasonality
seasonal = [np.sin(2*np.pi*t/12) +
            0.2*np.sin(2*np.pi*2*t/12) +
            0.1*np.sin(2*np.pi*4*t/12) +
            0.1*np.cos(2*np.pi*4*t/12)
            for t in time]
# putting together white noise + trend + seasonality
series = np.array(trend) + np.array(seasonal) + np.array(w)
plt.plot(series)
plt.show()
################################################################################

######################## non-linear models #####################################
# creating a seed
np.random.seed(1)
# mean and standard deviation
mu, sigma, length = 100, 20, 100 
# generating a gaussian white noise
w = np.random.normal(mu, sigma, length)
# generating the lags
time = range(length)
# generating a initial serie
z = [0.7 * time[t-1] + w[t] for t in time]
# generating a non-linear serie
serie = [np.exp(1+0.05*time[t])+z[t] for t in range(length)]
# plotting the non-linear time serie
plt.plot(serie)
plt.show()
################################################################################

######################## Logarithmic transformations ###########################
# to read the time series
serie = Series.from_csv('airline-passengers.csv', header=0)
serie.plot()
plt.show()

# acf on series
plot_acf(serie, lags=10)
plt.show()

# doing the transformation
serie_log = np.log(serie)
plt.plot(serie_log)
plt.show()

# acf on series
plot_acf(serie, lags=10)
plt.show()
################################################################################

######################## Forecasting from regression ###########################
# to read the time series
serie = Series.from_csv('airline-passengers.csv', header=0)
# log transformation
serie_log = np.log(serie)
serie_log.plot()
# train AR on serie with log transformation
model = AR(serie_log).fit()
# making the prediction
predictions = model.predict(start=len(serie_log), end=len(serie_log)+25, dynamic=False)
predictions.plot(label='AR Forecasting')
#to show
plt.legend()
plt.show()

# removing the log transformation on real serie
serie = np.exp(serie_log)
serie.plot()
# removing the log transformation on prediction
predictions = np.exp(predictions)
predictions.plot(label='AR Forecasting')
#to show
plt.legend()
plt.show()
################################################################################







