library(sspir)

# importing a time series
www <- 'http://www.massey.ac.nz/~pscowper/ts/Murray.txt'
Salt.dat <- read.table(www, header=T) ; attach(Salt.dat)


#simulating a time series
n <- 81 ; Time <- 1:n
SIN <- sin(2 * pi * Time /12)[-1]
COS <- cos(2 * pi * Time /12)[-1]
Chowilla <- Chowilla - mean(Chowilla)
Flow <- Flow - mean(Flow)
Chow <- Chowilla[-1]
Chow.L1 <- Chowilla[-n]
Flo <- Flow[-1]
Flo.L1 <- Flow[-n]
Sal.mat <- matrix(c(Chow, Flo), nrow = 80, ncol = 2)
x0 <- rep(1, (n-1))
xx <- cbind(x0, Chow.L1, Flo.L1, COS, SIN)
x.mat <- matrix(xx, nrow = n-1, ncol = 5)
G.mat <- diag(10)
W.mat <- diag(rep(c(10, 0.0001, 0.0001, 0.0001, 0.0001), 2))


# modeling 
m1 <- SS(y = Sal.mat, x = x.mat,
           Fmat = function(tt, x, phi) return (matrix(
               c(x[tt,1], x[tt,2], x[tt,3], x[tt,4], x[tt,5], rep(0,10),
                 x[tt,1], x[tt,2], x[tt,3], x[tt,4], x[tt,5]),
               nrow=10,ncol=2)),
           Gmat = function(tt, x, phi) return (G.mat),
           Wmat = function(tt, x, phi) return (W.mat),
           Vmat = function(tt, x, phi) return
           (matrix(c(839, -348, -348, 1612), nrow=2, ncol=2)),
           m0=matrix(c(0,0.9,0.1,-15,-10,0,0,0.7,30,20),nrow=1,ncol=10),
           C0 = 100 * W.mat)


# training a kalman filter
m1.f <- kfilter (m1)

# plotting
par(mfcol=c(2,3))
plot(m1.f$m[,1], type='l')
plot(m1.f$m[,2], type='l')
plot(m1.f$m[,3], type='l')
plot(m1.f$m[,6], type='l')
plot(m1.f$m[,7], type='l')
plot(m1.f$m[,8], type='l')

# plotting
par(mfcol=c(2,2))
plot(m1.f$m[,4], type='l')
plot(m1.f$m[,5], type='l')
plot(m1.f$m[,9], type='l')
plot(m1.f$m[,10], type='l')

