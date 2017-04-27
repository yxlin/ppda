rm(list=ls())
## length(x) == 1e4
## n == 1024+1
dat <- rnorm(100)
h <- 0.8*bw.nrd0(dat); h
z0 <- min(dat) - 3*h; z0
z1 <- max(dat) + 3*h; z1
x <- tnorm::rtn(1e4, 0, 1, z0, z1); str(x)

ngrid <- 2^10; ngrid
## seq(z0, z1, length.out=ngrid)
dt <- (z1-z0)/(ngrid-1)
z  <- z0+(0:(ngrid-1))*dt

## Get filter
tmp0 <- pi * ngrid/(z1-z0)
## seq(0, 1, length.out=1.0+0.5*ngrid)
dt <- 2/ngrid
tmp1 <- (0:(0.5*ngrid))*(2/ngrid)
freq <- tmp0 * tmp1
freq2 <- freq^2

h2 <- h^2
fil0 <- as.vector(exp(-0.5 * h2 * freq2))
length(fil0)
tmp <- as.matrix(fil0[2:(length(fil0)-1)], 1) ## 2nd to the one before the last
tmp3 <- pracma::flipud(tmp)


edges <- c(z - dt/2, z[1024] + dt/2)  



bin <- numeric(length(x))
cnt <- numeric(length(edges))
n <- length(edges)
## i is from 1 to 1024 (bin 1 to 1024); edges has 1025 elements
for (i in 1:(length(edges) - 1)) {
    li <- edges[i] <= x & x < edges[i + 1]
    cnt[i]  <- sum(li)
    bin[li] <- i  ## the index on simulation vector (yhat, simulated observations) 
                  ## belonging to bin i  
}


li <- x == edges[n]
cnt[n] <- sum(li)  ## get bin count 
bin[li] <- n


tmp1 <- cpda::histc(dat,edges)
length(tmp1)
tmp2 <- pracma::histc(dat,edges)
length(tmp2$cnt)
head(tmp2$bin)
plot(tmp1)
plot(tmp2$cnt)
