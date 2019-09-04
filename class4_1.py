'''
Created on 4 de set de 2019
@author: gusta
'''

# lib to use the acf and pcf correlations
from statsmodels.graphics.tsaplots import plot_pacf
# lib to import the arma methods
from statsmodels.tsa.arima_process import ArmaProcess
# lib to plot the data
import matplotlib.pyplot as plt
# to deal with numbers
import numpy as np
np.random.seed(0)

######################## simulating a model AR(1) ################################
#X_{t} = -1*x_{t-1} + z_{t}
# defining model parameters
ar = np.array([1, -1])
ma = np.array([1])
model = ArmaProcess(ar, ma)

# generating the new serie
serie = model.generate_sample(nsample=1000)

# plotting the serie
plt.plot(serie)
plt.show()

# to use the pacf
plot_pacf(serie, lags=10)
plt.show()
##################################################################################


######################## simulating a model ARMA(0,2) ############################
#X_{t} = 0.3*z_{t-1} + 0.7*z_{t-2}
# defining model parameters
ar = np.array([1])
ma = np.array([1, 0.3, 0.7])
model = ArmaProcess(ar, ma)

# generating the new serie
serie = model.generate_sample(nsample=1000)

# analyzing the series generated
print(model.isinvertible)
print(model.isstationary)

# plotting
plt.plot(serie)
plt.show()

# to use the pacf
plot_pacf(serie, lags=10)
plt.show()
#################################################################################


######################## simulating a model ARMA(2,2) ############################
#X_{t} = -1*x_{t-1} + 0.3*z_{t-1} + 0.7*z_{t-2}
# defining model parameters
ar = np.array([1, -1])
ma = np.array([1, 0.3, 0.7])
model = ArmaProcess(ar, ma)

# generating the new serie
serie = model.generate_sample(nsample=1000)

# plotting
plt.plot(serie)
plt.show()

# to use the pacf
plot_pacf(serie, lags=10)
plt.show()
#################################################################################
