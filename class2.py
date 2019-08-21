'''
Created on 21 de ago de 2019
@author: gusta
'''

# lib to handle with time series
from pandas import Series
# lib to plot the data
import matplotlib.pyplot as plt
# lib to import the stocastic basic models
from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing
# lib to import the ar model
from statsmodels.tsa.ar_model import AR

# to read the time series
serie = Series.from_csv('airline-passengers.csv', header=0)
serie.plot()

#################Simple Exponential#########################
# fit the data
ses = SimpleExpSmoothing(serie).fit(smoothing_level=0.8, optimized=True)

# plot the data
ses.fittedvalues.plot(label='SES Fit')
ses.forecast(25).rename('SES Forecasting').plot()

# to show the plot
plt.legend()
plt.show()
#############################################################

#################Holt and Winter#############################
# fit the data
hw = Holt(serie).fit(smoothing_level=0.8, smoothing_slope=0.9)

# plot the data
hw.fittedvalues.plot(label='HW Fit')

# to show the plot
hw.forecast(25).rename('HW Forecasting').plot()
plt.legend()
plt.show()
#############################################################


#################Exponential Smoothing#######################
# fit the data
es = ExponentialSmoothing(serie, 
                          seasonal_periods=12, 
                          trend='add', 
                          seasonal='add', 
                          damped=True).fit(use_boxcox=True)

# plot the data
es.fittedvalues.plot(label='ES Fit')
es.forecast(25).rename('ES Forecasting').plot()

# to show the plot
plt.legend()
plt.show()
#############################################################

#################Auto Regressive#######################
# train AR
model = AR(serie)
model_fitted = model.fit()

# parameters used
print("lags used: ", model_fitted.k_ar)
print("parameters: ", model_fitted.params)

# plot the prediction
predictions = model_fitted.predict(start=model_fitted.k_ar, end=len(serie), dynamic=False)
predictions.plot(label='AR Fit')
predictions = model_fitted.predict(start=len(serie), end=len(serie)+25, dynamic=False)
predictions.plot(label='AR Forecasting')

#to show
plt.legend()
plt.show()
#############################################################

