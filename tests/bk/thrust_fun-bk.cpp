#include <iostream>        // includes, std & armadillo
#include <armadillo>
#include <cuda_runtime.h>  // includes, cuda's runtime & fft
#include <cufft.h>
#include <cufftXt.h>
#include <curand_kernel.h> // Device random API

#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <math_constants.h>
#include <thrust/for_each.h>
#include <thrust/reverse.h>
#include <thrust/execution_policy.h>

struct pow_functor
{
  double a;
  
  pow_functor(double a_) : a(a_) {}
  
  __host__ __device__
  double operator()(double x) const
  {
    return pow(x, a);
  }
};

struct exp_functor
{
  __host__ __device__
  double operator()(double x) const
  {
    return exp(x);
  }
};

/* -------------------------------------------------------------------------  
 GPU-PDA operations 
------------------------------------------------------------------------- */
thrust::device_vector<double>
getFilter_gpu (double m, double M, double neg_half_h2, int p, const int n)
{
  const int N_grid = 1<<p;
  double tmp0 = CUDART_PI * (double)N_grid / (M - m); 
  const int N1 = 1+N_grid/2;
  const int N2 = N_grid/2;
  
  thrust::device_vector<double> term1(N1), term2(N1), freq(N1), h2vec(N1), 
  fil0(N1), fil1(N2);
  
  term2 = linspace_gpu(0.0, 1.0, 1+N_grid/2); 
  thrust::fill(term1.begin(), term1.end(), tmp0);
  thrust::transform(term1.begin(), term1.end(), term2.begin(), freq.begin(), 
    thrust::multiplies<double>());
  thrust::transform(freq.begin(), freq.end(), freq.begin(), pow_functor(2.));
  thrust::fill(h2vec.begin(), h2vec.end(), neg_half_h2);
  thrust::transform(h2vec.begin(), h2vec.end(), freq.begin(), fil0.begin(), 
    thrust::multiplies<double>());
  thrust::transform(fil0.begin(), fil0.end(), fil0.begin(), exp_functor());
  thrust::reverse_copy(thrust::device, fil0.begin()+1, fil0.begin()+N2, fil1.begin());
  fil0.insert(fil0.end(), fil1.begin(), fil1.end() );
  return fil0;
}

thrust::device_vector<double>
  linspace_gpu(double start, double end, const int n)
  {
    double Dx = (end-start)/(double)(n-1);
    thrust::device_vector<double> out(n);
    
    thrust::transform(
      thrust::make_counting_iterator(start/Dx),
      thrust::make_counting_iterator((end+1.f)/Dx),
      thrust::make_constant_iterator(Dx),
      out.begin(),
      thrust::multiplies<double>());
    return out;
  }




thrust::device_vector<double> getFilter_gpu(double m, double M, double h, double p); 
thrust::device_vector<double> linspace_gpu(double start, double end, const int n);