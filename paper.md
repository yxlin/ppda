---
title: 'Probability Density Approximation using Graphics Processing Unit'
tags:
  - PDA
authors:
 - name: Yi-Shin Lin
   orcid: 0000-0002-2454-6601
   affiliation: 1
affiliations:
 - name: University of Tasmania, Australia
   index: 1
date: 30 June 2017
bibliography: inst/bib/paper.bib
---

# Summary 

_gpda_ is an R package, conducting probability density approximation (PDA) 
[@Turner2014; @Holmes2015].  This package provides R functions and CUDA C API 
to harness the parallel computing power of graphics processing unit (GPU), 
making PDA computation efficient. Current release, version 0.18, mainly 
provides,

  * CUDA C API, which allow C programmers to construct their own 
  probability density approximation routines for any biological or cognitive 
  models and,
  * R functions, which approximates two choice-response-time cognitive 
  models: Canonical linear ballistic accumulation and piecewise LBA 
  models [@Brown2008; @Holmes2016].  

PDA calculates likelihood even when their analytic functions are 
unavailable.  It allows psychologists and biologists to model computationally 
complex biological processes, which in the
past could only be approached by overly simplified models. PDA is however 
computationally demanding.  It requires a large number of Monte Carlo 
simulations to attain satisfactory precision of approximation. Monte Carlo 
simulations add a heavy computational burden on every step of PDA algorithm. 

We implement _gpda_, using Armadillo C++ and CUDA libraries in order to provide
a practical and efficient solution for PDA, which is ready to apply on 
Bayesian computation. _gpda_ enables parallel computation with millions of
threads using GPU and avoids moving large chunk of 
memories back and forth between the system and GPU memories. Hence, _gpda_ 
practically removes the computational burden involving large numbers (>1e6) of 
model simulations without suffering the limitation of memory bandwidth. This 
solution allows one to rapidly approximate probability densities with ultra-high 
precision. 

# References
