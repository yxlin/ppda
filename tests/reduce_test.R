rm(list=ls())
## min 512 = 2^9 elements
2^10
dat0   <- gpda::rlba(2^11, nthread=64); str(dat0)
result <- gpda::min(dat0$RT); result
result <- gpda::max(dat0$RT); result
result <- gpda::minmax(dat0$RT); result
result <- gpda::sum(dat0$RT); result
result <- gpda::sd(dat0$RT); result

n <- 2^23
dat0   <- gpda::rlba(n, nthread=64); str(dat0)
result <- gpda::sqsum(dat0$RT); result
base::sum(dat0$RT^2);


stats::sd(dat0$RT);
base::min(dat0$RT); base::max(dat0$RT)
base::sum(dat0$RT);



dat0 <- cpda::rlba_test(2^21, b=1, A=.5, mean_v=c(2.4, 1.6), sd_v=c(1,1), t0=.5)
str(dat0)



variance <- sum((dat0$RT - mean(dat0$RT))^2) / length(dat0$RT)
sqrt(variance); variance


( sum(dat0$RT^2) - (sum(dat0$RT)^2) / length(dat0$RT) ) / 511
    


(dat0$RT - mean(dat0$RT))^2

sum( (dat0$RT - mean(dat0$RT))^2 )

round((dat0$RT[1:10] - mean(dat0$RT))^2, 4)

head(dat0)






res <- microbenchmark::microbenchmark(gpda::reduce0(dat0$RT),
                               gpda::reduce1(dat0$RT),
                               times=10L)
res

unsigned int tid = threadIdx.x;
unsigned int i   = blockIdx.x * blockDim.x + threadIdx.x;
unsigned int tid = threadIdx.x;
unsigned int i   = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

0*256+0:255
1*256+0:255

0*(256*2) + 0:255
1*(256*2) + 0:255

0*(256*2) + 0:255
##1*(256/2*2) + 0:255

tid <- 0:7
nBlk <- 16
nTh <- 8
tmp <- matrix(numeric(nBlk*nTh), ncol=nTh)
for(i in 0:(nBlk-1)) {
    tmp[i+1,] <- i * nBlk + (0:(nTh-1))
}
tmp


