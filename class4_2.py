'''
Created on 4 de set de 2019
@author: gusta
'''

######################## libs that will be used ################################
# lib to plot the data
import matplotlib.pyplot as plt
# lib to handle with time series
import pandas as pd
# to deal with numbers
import numpy as np
################################################################################

####################### making the connection with R and Python ################
# defining the variables path in the system
import os
os.environ['R_HOME'] = 'C:/Program Files/R/R-3.3.0'
os.environ['R_USER'] = 'C:/Users/gusta/Anaconda3/Lib/site-packages/rpy2'

# activating the connection
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

# importing a R lib
from rpy2.robjects.packages import importr 
forecast = importr("forecast")
################################################################################


################### data treatment #############################################
# to read the time series
serie = pd.read_csv('airline-passengers.csv')["Passengers"].values

# spliting the series in training and test
training, test = serie[:80], serie[80:]
################################################################################

####################### ARIMA modeling #########################################
# training a ARIMA
modelFitted = forecast.auto_arima(training)
#modelFitted = forecast.Arima(training, order=np.array([2,1,0]), seasonal=np.array([1,1,0]))
# making a prediction
prediction = forecast.fitted_Arima(modelFitted)

#plotting 
plt.plot(training, label="training")
plt.plot(prediction, label="prediction")
plt.legend()
plt.show()
################################################################################


####################### ARIMA forecasting ######################################
# presenting the test set and the fitted model
model = forecast.Arima(test, model=modelFitted)
# making the forecasting
prediction = forecast.fitted_Arima(model)

#plotting 
plt.plot(test, label="test")
plt.plot(prediction, label="prediction")
plt.legend()
plt.show()
################################################################################
