'''
Created on 21 de ago de 2019
@author: gusta
'''

# lib to handle with time series
from pandas import Series
# lib to plot the data
import matplotlib.pyplot as plt
# lib to implort the wolt winter model
from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing
from statsmodels.tsa.ar_model import AR

# to read the time series
serie = Series.from_csv('airline-passengers.csv', header=0)
serie.plot()

#################Simple Exponential#########################
ses = SimpleExpSmoothing(serie).fit(smoothing_level=0.8, optimized=True)
#ses.fittedvalues.plot(label='SES Fit')
ses.forecast(25).rename('SES Forecasting').plot()
#############################################################

#################Holt and Winter#############################
hw = Holt(serie).fit(smoothing_level=0.8, smoothing_slope=0.9)
#hw.fittedvalues.plot(label='HW Fit')
hw.forecast(25).rename('HW Forecasting').plot()
#############################################################

#################Exponential Smoothing#######################
es = ExponentialSmoothing(serie, seasonal_periods=12, trend='add', seasonal='add', damped=True).fit(use_boxcox=True)
#es.fittedvalues.plot(label='ES Fit')
es.forecast(25).rename('ES Forecasting').plot()
#############################################################

#################Auto Regressive#######################
model = AR(serie)
model_fitted = model.fit()
print("lags used: ", model_fitted.k_ar)
print("parameters: ", model_fitted.params)

predictions = model_fitted.predict(start=len(serie), end=len(serie)+25, dynamic=False)
predictions.plot(label='AR Forecasting')
#############################################################

#to show plot
plt.legend()
plt.show()

