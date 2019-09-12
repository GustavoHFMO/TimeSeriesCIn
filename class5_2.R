library(sspir)

# creating a seed
set.seed(1)

# synthetic series
x1 <- 1:30
x1 <- x1/10 + 2
a <- c(rep(4,15), rep(5,15))
b <- c(rep(2,15), rep(-1,15))
n <- length(x1)
y1 <- a + b * x1 + rnorm(n)
x0 <- rep(1, n)
xx <- cbind(x0, x1)
x.mat <- matrix(xx, nrow = n, ncol = 2)
y.mat <- matrix(y1, nrow = n, ncol = 1)
m1 <- SS(y = y.mat, x = x.mat,
           Fmat = function(tt,x,phi)
             return( matrix(c(x[tt,1], x[tt,2]), nrow = 2, ncol = 1)),
           Gmat = function(tt,x,phi) return (diag(2)),
           Wmat = function(tt, x, phi) return (0.1*diag(2)),
           Vmat = function(tt,x,phi) return (matrix(1)),
           m0 = matrix(c(5,3),nrow=1,ncol=2),C0=10*diag(2))

# plotting the synthetic serie
plot(m1$y, ylab = "Closing price", main = "Simulated")

# kalman filter
m1.f <- kfilter(m1)

# plotting
par(mfcol=c(2,1))
plot(m1.f$m[,1], type='l')
plot(m1.f$m[,2], type='l')

