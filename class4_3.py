'''
Created on 4 de set de 2019
@author: gusta
'''

############################################# libs ###############################
# lib to import the arma methods
from statsmodels.tsa.arima_process import ArmaProcess
# lib to plot the data
import matplotlib.pyplot as plt
# importing the lib for garch
from arch import arch_model
# importing lib for numbers
import numpy as np
np.random.seed()
##################################################################################

######################## simulating a model AR(2) ############################
#X_{t} = -0.2*x_{t-1} + 0.8*x_{t-2} + z_{t}
# defining model parameters
ar = np.array([1, -0.3, 0.8])
ma = np.array([1])
model = ArmaProcess(ar, ma)

# generating the new serie
serie = model.generate_sample(nsample=300)

# plotting
plt.plot(serie)
plt.show()
#################################################################################


############################# GARCH model #######################################
# training a Garch model
garch = arch_model(serie, vol='GARCH', p=2, q=2).fit( disp='off')

# printing the statisticals
print(garch.summary)

# plotting the modelling
plt.plot(serie, label='Real')
plt.plot(garch._volatility, label = '+Vol')
plt.plot(-garch._volatility, label = '-Vol')
plt.legend(loc='best')
plt.show()
#################################################################################

