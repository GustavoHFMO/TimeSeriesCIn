# importing the lib
library(sspir)

# creating a seed
set.seed(1)

# creating a synthetic serie
Plummet.dat <- 20 + 2*rnorm(20) + c(rep(0,10), rep(-10,10))
n <- length(Plummet.dat)
Plummet.mat <- matrix(Plummet.dat, nrow = n, ncol = 1)
m1 <- SS(y = Plummet.mat,
           Fmat = function(tt,x,phi) return( matrix(1) ),
           Gmat = function(tt,x,phi) return( matrix(1) ),
           Wmat = function(tt,x,phi) return( matrix(0.1)),
           Vmat = function(tt,x,phi) return( matrix(2) ),
           m0 = matrix(25), C0 = matrix(10))

# plotting the synthetic serie
plot(m1$y, ylab = "Closing price", main = "Simulated")

# training a kalman filter
m1.f <- kfilter(m1)
m1.s <- smoother(m1.f)

# plotting the predictions
lines(m1.f$m, lty = 2)
lines(m1.s$m, lty = 3)

