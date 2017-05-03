#include <iostream>        // includes, standard template & armadillo library
#include <armadillo>
#include <cuda_runtime.h>  // includes, cuda's runtime & fft
#include <cufft.h>
#include <cufftXt.h>
#include <curand_kernel.h> // Device random API
#include <ctime> // CPU timer
//#include "../inst/include/constant.h"  // math constants
#include "../inst/include/random.h"
#include "../inst/include/density.h"  

extern "C" void logLik_fft2(double *y_, double *yhat_, int *ny, 
                            int *ns, int *p, int *nThreads, double *out);

unsigned int nextPow2(unsigned int x) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}


/* -------------------------------------------------------------------------  
 KDE operations 
------------------------------------------------------------------------- */
arma::vec getEdges(arma::vec z, double dt)
{
  arma::vec term1 = z - 0.5*dt;
  arma::vec term2(1) ;
  term2.fill(z[z.size()-1] + 0.5*dt) ;
  return arma::join_cols(term1, term2);
}

arma::vec getFilter(double m, double M, double h, double p) {
  // cannoical Gaussian kernel
  double tmp0    = 2 * arma::datum::pi * (std::pow(2, p) / (M-m)) * 0.5;
  arma::vec tmp1 = arma::linspace<arma::vec>(0, 1, 1 + (std::pow(2, p)/2));
  arma::vec freq = tmp0 * tmp1 ;
  arma::vec s2   = arma::pow(freq, 2) ; // s^2 on p17
  double h2      = h * h;
  arma::vec fil0 = arma::exp(-0.5 * h2 * s2) ;
  arma::vec fil1 = arma::flipud(fil0.rows(1, (fil0.size() - 2)));
  arma::vec out  = arma::join_cols(fil0, fil1) ;
  return out ;
}

arma::vec pmax(arma::vec v, double min)
{
  for (arma::vec::iterator it=v.begin(); it!=v.end(); it++)
  {
    if (*it < min) *it = min ;
  }
  return v ;
}

arma::vec getVec(double *x, int *nx)
{
  arma::vec out(*nx);
  for(int i=0; i<*nx; i++) { out[i]=*(x+i); }
  return out;
}

arma::vec density(arma::vec y, arma::vec be, double dt)
{
  // y is yhat; be is binEdges; ns is nSamples
  arma::uvec hc       = arma::histc(y, be) ;
  arma::vec bincount  = arma::conv_to<arma::vec>::from(hc);
  int ns              = arma::accu(bincount);
  arma::vec PDF_hist  = bincount / (dt * ns);
  arma::vec out       = PDF_hist.rows(0, (PDF_hist.size() - 2)) ;
  return out ;
}

double cquantile(arma::vec y, double q)
{
  arma::vec sy = sort(y);
  int nth = sy.n_elem*(q - (1e-9));
  return sy(nth);
}

double bwNRD0(arma::vec y, double m)
{ // y must be a simulation vector
  // double h   = (q75-q25)/1.34 ; // R divides 1.34
  int n = y.n_elem ;
  return m*0.9*std::min((cquantile(y, 0.75) - cquantile(y, 0.25)),
                      arma::stddev(y))*std::pow((double)n, -0.2);
}

double gaussian(double y, arma::vec yhat, double h) {
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
  return ( arma::sum(result) / ns);
}

void logLik_fft2(double *y_, double *yhat_, int *ny, int *ns,  
                 int *p, int *nThread, double out[])
{
    double *d_rngnum,*h_rngnum, *d_mean, *h_mean, *d_sd, *h_sd;
    h_mean = (double *)malloc(1 * sizeof(double));
    h_sd   = (double *)malloc(1 * sizeof(double));
    cudaHostAlloc((void**)&h_rngnum, *ns * sizeof(double), cudaHostAllocDefault);
    h_mean[0] = 0;
    h_sd[0]   = 1;
    int nBlk  = (*ns)/(*nThread) + 1;

    cudaMalloc((void**) &d_mean, 1 * sizeof(double));
    cudaMalloc((void**) &d_sd,   1 * sizeof(double));
    cudaMalloc((void**) &d_rngnum,  *ns * sizeof(double));
    cudaMemcpy(d_mean, h_mean,   1 * sizeof(double), cudaMemcpyHostToDevice);
    rnorm_kernel<<<nBlk, *nThread>>>(*ns, d_mean, d_sd, d_rngnum);
    cudaMemcpy(h_rngnum, d_rngnum, *ns * sizeof(double), cudaMemcpyDeviceToHost);

    arma::vec yhat = getVec(h_rngnum, ns);

    double h  = bwNRD0(yhat, 1.0);
    double z0 = yhat.min() - 3 * h;
    double z1 = yhat.max() + 3 * h;
    int ngrid = 1<<*p;
    int ngridplus1 = ngrid + 1;
    int ngridminus1 = ngrid - 1;
    double dt = (z1-z0)/ngridminus1;
    arma::vec z = arma::linspace<arma::vec>(z0, z1, ngrid);
    arma::vec y = getVec(y_, ny); // Convert to arma vec

  /*--------------------------------------------
    Calculate the location of bin edges. (1025)
    -------------------------------------------*/
  double *h_binEdges;
  cudaHostAlloc((void**)&h_binEdges, ngridplus1 * sizeof(double), cudaHostAllocDefault);
  for(int i=0; i<ngridplus1; i++)
  {
    h_binEdges[i] = i < ngrid ? z0 + dt*((double)i - 0.5) : (z0 + (double)(i-1)*dt) +  0.5*dt;
  }

  /*--------------------------------------------
  Calculate Gaussian filter at frequency domain. (1024)
  -------------------------------------------*/
  arma::vec filter(ngrid);
  double z1minusz0 = z1 - z0;
  int halfngrid = 1<<(*p-1);
  double fil0_constant = -2*h*h*M_PI*M_PI/(z1minusz0 * z1minusz0);
  for(int i=0; i < ngrid; i++) {
           if (i < (1+halfngrid)) {
               filter[i] = std::exp(fil0_constant * i*i);
           } else { // magical flipping
               int j = 2*(i - halfngrid);
               filter[i] = filter[i-j];
           }
   }

  /*--------------------------------------------
  Calculate histogram (1024)
  -------------------------------------------*/
  unsigned int *d_histo;
  unsigned int *h_histo;
  double *d_binEdges;

  cudaHostAlloc((void**)&h_histo, ngrid * sizeof(unsigned int), cudaHostAllocDefault);

  for(int i=0; i<ngrid; i++) {h_histo[i] = 0;}

  cudaMalloc((void**) &d_binEdges,  ngridplus1*sizeof(double));
  cudaMalloc((void**) &d_histo,  ngrid*sizeof(unsigned int));
  cudaMemcpy(d_binEdges, h_binEdges, ngridplus1*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_histo, h_histo, ngrid*sizeof(unsigned int), cudaMemcpyHostToDevice);
  
  histc_kernel<<<nBlk, *nThread>>>(d_binEdges, d_rngnum, ns, d_histo);

  cudaMemcpy(h_histo, d_histo, sizeof(unsigned int)*ngrid, cudaMemcpyDeviceToHost);

  arma::vec signal(ngrid);
  for(int i=0; i<ngrid; i++)
  {
      signal[i] = h_histo[i] / (dt* (*ns));
  }

  cudaFreeHost(h_histo);
  cudaFreeHost(h_binEdges);

  arma::vec PDF_cpp;
  arma::vec PDF0_cpp = arma::real(arma::ifft(filter % arma::fft(signal))) ; // smoothed
  arma::interp1(z, PDF0_cpp, y, PDF_cpp);
  arma::vec PDF_tmp_cpp = pmax(PDF_cpp, std::pow(10, -10)) ;

  for(arma::vec::iterator i=PDF_tmp_cpp.begin(); i!=PDF_tmp_cpp.end(); ++i)
  {
    int idx  = std::distance(PDF_tmp_cpp.begin(), i);
    out[idx] = std::log(*i);
  }
}
