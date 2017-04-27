#include <stdio.h>
#include <iostream> // Debug printing
#include <assert.h>
#define MATHLIB_STANDALONE 
#include "Rmath.h"
#include <armadillo>

/*
extern "C" void dexG(int *n, double *x_, double *mu, double *sigma, double *tau, 
                     int *nThreads, double *out);

arma::vec getVec(double *x, int *n) 
{
  arma::vec out(*n);
  for(int i=0; i<*n; i++) { out[i]=*(x+i); }
  return out;
}

void dexG(int *n, double *x_, double *mu, double *sigma, double *tau, 
          int *nThreads, double *out)
{
  arma::vec x = getVec(x_ , n);
  double density, term1, term2;
  for(arma::vec::iterator it=x.begin(); it!=x.end(); ++it)
  {
    int i = std::distance(x.begin(), it);
    term1 = (*mu - *it) / *tau + (*sigma) * (*sigma) / (2*(*tau));
    //term2 = R::pnorm((*it - *mu) / *sigma - *mu / *tau, 0, 1, 1, 0)
  }

}
*/

/*
double exGaussian(double y, arma::vec yhat, double h) {
  // standard gaussian kernel mean=0; sigma==1
  double x;
  int ns = yhat.n_elem;
  arma::vec result(ns);
  
  for(arma::vec::iterator it=yhat.begin(); it!=yhat.end(); ++it)
  {
    int i = std::distance(yhat.begin(), it);
    x = (y - *it)/h;  // z / h
    // (1/h) * K(z/h); K_h(z)
    result[i] = ( (1/(sqrt(2*arma::datum::pi))) * exp( -pow(x,2) / 2 ) ) / h;
  }
  // (1/N_s) * sigma K_h (x-x.tidle_j)
  return ( arma::accu(result) / ns);
}
*/
