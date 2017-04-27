require(gpda)
answer <- .C("runif_gpu", as.integer(10), as.double(0), as.double(1), 
             numeric(10), PACKAGE='gpda')

answer <- .C("rnorm_gpu", as.integer(10), as.double(0), as.double(1), 
             numeric(10), 
             PACKAGE='gda')



rt1 <- rLBA(500, A=0.5, b=1, t0 = 0.5, mean_v=c(2.4, 1.6), sd_v=c(1,1.2))

answer <- .C("rlba_gpu", as.integer(10), as.double(1), as.double(1), 
             as.double(0.5), as.double(0.5), as.double(2.4), as.double(3.6),
             as.double(1), as.double(2.2), as.double(.5), as.double(.5),
             as.integer(16), numeric(10), integer(10),
             PACKAGE='gda')
answer[[13]]
answer[[14]]

gda::rlba(n=10, b=1, A=.5, mean_v=c(2.4, 1.6), sd_v=c(1, 1.2), t0=.5)
rtdists::rLBA(10, A=0.5, b=1, t0 = 0.5, mean_v=c(2.4, 1.6), 
                     sd_v=c(1,1.2), silent = T)


system.time(rt1 <- gda::rlba(n=1e5, b=1, A=.5, mean_v=c(2.4, 1.6), 
                            sd_v=c(1, 1.2), t0=.5, nThreads=8))
head(rt1)
pVec <- c(b1=1, b2=1, A1=.5, A2=.5, mu1=2.4, mu2=1.6, sigma1=1, sigma2=1.2,
          t01=.5, t02=.5)
system.time(rt2 <- pda::rlba(n = 1e5, pVec = pVec))
head(rt2[[1]])
head(rt2[[2]])

nin <- 2e7
require(rbenchmark)
within(benchmark(gpu=gda::rnorm(nin),
                 cpu=stats::rnorm(nin),
                 replications=rep(10, 3),
                 columns=c('test', 'replications', 'elapsed'),
                 order=c('test', 'replications')),
       { average = elapsed/replications })

