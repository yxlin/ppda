# Probability Density Approximation using Graphic Processing Unit 

The package uses GP-GPU to conduct Monte Carlo simulations of 
cannoical and piece-wise linear ballistic accumulator models. The 
simulations are then used to approximate the model likelihood via a slihgtly
modified method based on Homles (2015). 

## Getting Started

```
require(gpda)


```

## Installation 

```
## From github
devtools::install_github("TasCL/gpda")
## From source: 
install.packages("gpda_0.1.3.tar.gz", repos = NULL, type="source")
```

## Prerequisites
 - R (>= 3.0.2)
 - nvcc, gcc & g++ (>= 4.4)
 - Nvidia GPU card Compute Capability (>= 2.0)
 - Nvidia's CUDA toolkit (>= release 7.5)
 - [Armadillo](http://arma.sourceforge.net/download.html) (>= 5.100)
  
## Contributors

- Yi-Shin Lin <yishin.lin@utas.edu.au> 
- Andrew Heathcote 
- William Holmes 

## References
* Holmes, W. (2015). A practical guide to the Probability Density
Approximation (PDA) with improved implementation and error characterization.
Journal of Mathematical Psychology, 68-69, 13--24,
doi: http://dx.doi.org/10.1016/j.jmp.2015.08.006.

## Acknowledgments
* gpda R packaging is based on gputools 1.1.
