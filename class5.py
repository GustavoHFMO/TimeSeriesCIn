'''
Created on 25 de out de 2017
@author: gusta
'''

# to plot the data
import matplotlib.pyplot as plt
# lib to read the data
import pandas as pd

####################### making the connection with R and Python ################
# defining the variables path in the system
import os
os.environ['R_HOME'] = 'C:/Program Files/R/R-3.3.0'
os.environ['R_USER'] = 'C:/Users/gusta/Anaconda3/Lib/site-packages/rpy2'

# importing a R lib
from rpy2.robjects.packages import importr 
sspir = importr("sspir")

# activating the connection to numpy
from rpy2.robjects import numpy2ri
numpy2ri.activate()
################################################################################

################### data treatment #############################################
# to read the time series
serie = pd.read_csv('airline-passengers.csv')["Passengers"].values

# spliting the series in training and test
training, test = serie[:80], serie[80:]
################################################################################

####################### Kalman Filter modeling #########################################
# training a Kalman Filter
modelFitted = sspir.kfilter(training)
# making a prediction
prediction = sspir.smoother(modelFitted)

#plotting 
plt.plot(training, label="training")
plt.plot(prediction, label="prediction")
plt.legend()
plt.show()
################################################################################


