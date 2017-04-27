#include <unistd.h> 
#include <stdio.h>  // C printing
#include <iostream> // C++ printing Debug printing
#include <assert.h> // C check
#include <math.h>

#include <curand.h> // Host random API
#include <curand_kernel.h> // Device random API

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
printf("Error at %s:%d\n",__FILE__,__LINE__);    \
return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
printf("Error at %s:%d\n",__FILE__,__LINE__);                \
return EXIT_FAILURE;}} while(0)

#define SQRT_2PI   2.5066282746310007e+0 /* sqrt(2 x pi) */
#define M_E	   2.7182818284590452354	/* e */
#define CUDART_INF              __longlong_as_double(0x7ff0000000000000ULL) /* work insider device */

extern "C" void runif_gpu(int *n, double *min, double *max, int *nThreads, double *out);
extern "C" void rnorm_gpu(int *n,  double *mean, double *sd, int *nThreads, double *out);
extern "C" void rtnorm_gpu(int *n, double *mean, double *sd, double *lower, double *upper,
                            int *nThreads, double *out);

extern "C" void rlba_gpu(int *n, double *b, double *A, double *v, int *nv,
                         double *sv, double *t0, int *nThreads, double *RT, int *R);


/* this GPU kernel function is used to initialize the random states */
__global__ void init(unsigned int seed, curandState_t* states, int n) {
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < n; i += numThreads)
    {
          curand_init(seed, i, 0, &states[i]);
    }
}

__global__ void randoms_uniform(curandState_t* states, double* numbers, int n, double* min, double* max) {
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < n; i += numThreads)
    {
       numbers[i] = curand_uniform_double(&states[i]);
       numbers[i] *= (max[0] - min[0] + 0.0000001);
       numbers[i] += min[0];
    }
}

__global__ void randoms_normal(curandState_t* states, double* numbers, int n, double* mean, double* sd) {
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < n; i += numThreads)
    {
        numbers[i] = curand_normal_double(&states[i]);
        numbers[i] += mean[0];
        numbers[i] *= sd[0];
    }
}

// Accept-Reject Algorithm 0;
__global__ void rtnorm0(unsigned int seed, curandState_t* state, int n,
                        double* mean, double* sd, double* lower,
                        double* upper, double* out) {
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    int valid;
    double z;
    for (int i = threadID; i < n; i += numThreads)
    {
        valid = 0;
        curand_init(seed, i, 0, &state[i]);
        while (valid == 0)
        {
           z = curand_normal_double(&state[i]);
           if (z <= upper[0] && z >= lower[0])
           {
               out[i] = z * sd[0] + mean[0];
               valid  = 1 ;
           }
        }
    }
}

// Accept-Reject Algorithm 0;
static __device__ inline double rtnorm0_device(int threadID, curandState_t* state,
                                               double a, double mean, double sd, 
                                               double lower, double upper) {
    double z, out; // a stands for alphaStar in Robert (1995)
    int valid = 0;

    curand_init(clock64(), threadID, 0, &state[threadID]);
    while (valid == 0)
    {
         z = curand_normal_double(&state[threadID]);
         if (z <= upper && z >= lower)
         {
              out = z * sd + mean;
              valid = 1;
         }
    }
   return out;
}


// Algorithm 2; 'expu'; use when upper < mean; lower = -INFINITY.
__global__ void rtnorm2(unsigned int seed, curandState_t* state, int n,
                        double* mean, double* sd, double* lower,
                        double* upper, double* out) {
     const int numThreads = blockDim.x * gridDim.x;
     const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
     int valid = 0;
     double z = 0, r = 0, u = 0; // a stands for alphaStar in Robert (1995)
     double a = 0.5 * (sqrt(upper[0] * upper[0] + 4.0) - upper[0]);

     for (int i = threadID; i < n; i += numThreads)
     {
         valid = 0;
         curand_init(seed, i, 0, &state[i]);
         while (valid == 0)
         {
            /* rexp via runif. a=0, b=1/alphaStar; see Saucier (2000)
              assert( b > 0.);
              return a - b * log( uniform(0., 1.)) */
             z = (-1/a) * log(curand_uniform_double(&state[i])) - upper[0];
             u = curand_uniform_double(&state[i]);
             r = exp(-0.5 * (z - a) * (z - a));
             if (u <= r && z <= -lower[0])
             {
                out[i] = -z * sd[0] + mean[0]; // note '-' before z
                valid = 1;
             }
         }
     }
}


// Algorithm 1; 'expl'; use when lower > mean; upper = INFINITY
__global__ void rtnorm1(unsigned int seed, curandState_t* state, int n,
                        double* mean, double* sd, double* lower,
                        double* upper, double* out) {
     const int numThreads = blockDim.x * gridDim.x;
     const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
     int valid = 0;              // alphaStar  must be greater than 0
     double z = 0, r = 0, u = 0; // a stands for alphaStar in Robert (1995)
     double a = 0.5 * (sqrt(lower[0] * lower[0] + 4.0) + lower[0]); 

     for (int i = threadID; i < n; i += numThreads)
     {
         valid = 0;
         curand_init(seed, i, 0, &state[i]);
         while (valid == 0)
         {
           // rexp via runif. a=0, b=1/alphaStar; see Saucier (2000)
           //   assert( b > 0.);
           //   return a - b * log( uniform(0., 1.)) 
           z = (-1/a) * log(curand_uniform_double(&state[i])) + lower[0];
           u = curand_uniform_double(&state[i]);
           r = exp(-0.5 * (z - a) * (z - a));
           if (u <= r && z <= upper[0])
           {
             out[i] = z * sd[0] + mean[0];
             valid = 1;
           }
         }
     }
}


// Algorithm 1; 'expl'; use when lower > mean; upper = INFINITY
static __device__ inline double rtnorm1_device(int threadID, curandState_t* state,
                                               double a, double mean, double sd, 
                                               double lower, double upper) {
    double z, r, u, out; // a stands for alphaStar in Robert (1995)
    int valid = 0;

    curand_init(clock64(), threadID, 0, &state[threadID]);
    while (valid == 0)
    {
         
         z = (-1/a) * log(curand_uniform_double(&state[threadID])) + lower;
         u = curand_uniform_double(&state[threadID]);
         r = exp(-0.5 * (z - a) * (z - a));
         if (u <= r && z <= upper)
         {
              out = z * sd + mean;
              valid = 1;
         }
    }
   return out;
}

// Algorithm 3; u;  page 123. 2.2. Two-sided truncated normal dist.
__global__ void rtnorm3(unsigned int seed, curandState_t* state, int n,
                        double* mean, double* sd, double* lower,
                        double* upper, double* out) {
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
    int valid = 0;
    double z = 0, r = 0, u = 0;
    double maxminusmin = upper[0] - lower[0] + 1e-7;
    double lower2 = lower[0] * lower[0];
    double upper2 = upper[0] * upper[0];

    for (int i = threadID; i < n; i += numThreads)
    {
        valid = 0;
        curand_init(seed, i, 0, &state[i]);
        while (valid == 0)
        {
            z = lower[0] + maxminusmin * curand_uniform_double(&state[i]);
            if (lower[0] > 0) {
                r = exp( 0.5 * (lower2 - z*z) );
            } else if (upper[0] < 0) {
                r = exp( 0.5 * (upper2 - z*z) );
            } else { 
                r = exp( -0.5 * z * z );
            }
            u = curand_uniform_double(&state[i]);
            if (u <= r)
            {
             out[i] = z * sd[0] + mean[0];
             valid = 1;
            }
        }
    }
}


__global__ void rlba(curandState_t* state, int* n, double* b,
                     double* A, double* v, double* sv,
                     double* t0, double* lower, double* upper, double* alphaStar, double* RT, int* R) {
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
    double x01, x02, rate1, rate2, dt1, dt2;

    // For rlba to use rtnorm, there are only two possibilities: 
    // algorithm 0 or algorithm1. So I use a boolean switch for algorithm 0.
    // If the boolean return true (1), rlba will use Naive A-R method (alg0).
    // Otherwise, it will use left-trucation method (alg1, 'expl') 

    // Algorithm (0): Use Naive A-R method for choice 1
    bool choice1 = (lower[0] < 0 && upper[0]==CUDART_INF) || (lower[0]==-CUDART_INF && upper[0]) ||
            (isfinite(lower[0]) && isfinite(upper[0]) && lower[0] < 0 && upper[0] > 0 && ((upper[0] - lower[0]) > SQRT_2PI));

    // Algorithm (0): Use Naive A-R method for choice 2
    bool choice2 = (lower[1] < 0 && upper[1]==CUDART_INF) || (lower[1]==-CUDART_INF && upper[1]) ||
            (isfinite(lower[1]) && isfinite(upper[1]) && lower[1] < 0 && upper[1] > 0 && ((upper[1] - lower[1]) > SQRT_2PI));


    for (int i = threadID; i < n[0]; i += numThreads)
    {
        curand_init(clock64(), i, 0, &state[i]);
        x01 = A[0] * curand_uniform_double(&state[i]);
        x02 = A[0] * curand_uniform_double(&state[i]);
        rate1 = (choice1) ? rtnorm0_device(i, &state[i], alphaStar[0], v[0], sv[0], lower[0], upper[0]) :
                rtnorm1_device(i, &state[i], alphaStar[0], v[0], sv[0], lower[0], upper[0]);
        rate2 = (choice2) ? rtnorm0_device(i, &state[i], alphaStar[1], v[1], sv[0], lower[1], upper[1]) :
                rtnorm1_device(i, &state[i], alphaStar[0], v[1], sv[0], lower[1], upper[1]);
        dt1 = (b[0] - x01) / rate1;
        dt2 = (b[0] - x02) / rate2;
        RT[i] = (dt1 < dt2) ? (dt1 + t0[0]) : (dt2 + t0[0]);
        R[i]  = (dt1 < dt2) ? 1 : 2;
    }
    
}


void runif_gpu(int *n, double *min, double *max, int *nThreads, double *out)
{
  
  curandState_t *d_states;        // Declare memory space in host side for   
  double *d_min, *d_max, *d_out;  // receiving GPU memory of the random seeds
  int nthreads = *nThreads ;      // Define the execution configuration
  int nblocks  = (*n)/(nthreads) + 1;

  // Allocate and initialize host/CPU memory 
  double *h_min = (double *)malloc(sizeof(double) * (1));
  double *h_max = (double *)malloc(sizeof(double) * (1));
  h_min[0] = *min;
  h_max[0] = *max;

  /* allocate space on the GPU for the random states and gpu rng */
  cudaMalloc((void**) &d_states, (*n) * sizeof(curandState_t));
  cudaMalloc((void**) &d_out,    (*n) * sizeof(double));
  cudaMalloc((void**) &d_min,    (1)  * sizeof(double));
  cudaMalloc((void**) &d_max,    (1)  * sizeof(double));

  // Copy host memory to device
  cudaMemcpy(d_min, h_min, sizeof(double)*(1), cudaMemcpyHostToDevice);
  cudaMemcpy(d_max, h_max, sizeof(double)*(1), cudaMemcpyHostToDevice);

  /* GPU initializes all of the random states and generates random #s*/
  init<<<nblocks, nthreads>>>(time(0), d_states, (*n));
  randoms_uniform<<<nblocks, nthreads>>>(d_states, d_out, (*n), d_min, d_max);

  /* copy the random numbers back */
  cudaMemcpy(out, d_out, (*n) * sizeof(double), cudaMemcpyDeviceToHost);

  free(h_min);
  free(h_max);
  cudaFree(d_states);
  cudaFree(d_out);
  cudaFree(d_min);
  cudaFree(d_max);
}

void rnorm_gpu(int *n, double *mean, double *sd, int *nThreads, double *out)
{
  curandState_t *d_states;  
  double *d_out, *d_mean, *d_sd;
  int nthreads = *nThreads;   // Define the execution configuration
  int nblocks  = (*n)/(nthreads) + 1;

  // Initialise the memory 
  double *h_mean = (double *)malloc(sizeof(double) * (1));
  double *h_sd   = (double *)malloc(sizeof(double) * (1));
  h_mean[0]      = *mean;
  h_sd[0]        = *sd;

  /* allocate space on the GPU for the random states and gpu rng */
  cudaMalloc((void**) &d_states, (*n) * sizeof(curandState_t));
  cudaMalloc((void**) &d_out,    (*n) * sizeof(double));
  cudaMalloc((void**) &d_mean,   (1)  * sizeof(double));
  cudaMalloc((void**) &d_sd,     (1)  * sizeof(double));

  cudaMemcpy(d_mean, h_mean, sizeof(double)*(1), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sd, h_sd,     sizeof(double)*(1), cudaMemcpyHostToDevice);

  init<<<nblocks, nthreads>>>(time(0), d_states, (*n));
  randoms_normal<<<nblocks, nthreads>>>(d_states, d_out, (*n), d_mean, d_sd);

  /* copy the random numbers back */
  cudaMemcpy(out, d_out, (*n) * sizeof(double), cudaMemcpyDeviceToHost);

  free(h_mean);
  free(h_sd);
  cudaFree(d_states);
  cudaFree(d_out);
  cudaFree(d_mean);
  cudaFree(d_sd);
}


void rtnorm_gpu(int *n, double *mean, double *sd, double *lower, double *upper, 
   int *nThreads, double *out)
{
  curandState_t *d_state;  
  double *d_out, *d_mean, *d_sd, *d_l, *d_u; // lower and upper bounds
  double *h_mean, *h_sd, *h_u, *h_l;
  int nthreads = *nThreads;   // Define the execution configuration
  int nblocks  = (*n)/(nthreads) + 1;

  h_mean = (double *)malloc(sizeof(double) * (1));
  h_sd   = (double *)malloc(sizeof(double) * (1));
  h_u    = (double *)malloc(sizeof(double) * (1));
  h_l    = (double *)malloc(sizeof(double) * (1));
  h_mean[0] = *mean;
  h_sd[0]   = *sd;
  h_l[0]    = (*lower - *mean) / *sd; // convert to mean=0, sd=1
  h_u[0]    = (*upper - *mean) / *sd;

  // Algorithm (0): Use Naive A-R method
  bool a0 = (h_l[0] < 0 && h_u[0]==INFINITY)  || (h_l[0] == -INFINITY && h_u[0] > 0) ||
      (std::isfinite(h_l[0]) && std::isfinite(h_u[0]) && h_l[0] < 0 && h_u[0] > 0 &&
      (h_u[0] - h_l[0]) > SQRT_2PI);

  // Algorithm (1): Use Proposition 2.3 with only lower truncation. upper==INFINITY
  // rejection sampling with exponential proposal. Use if lower > mean
  // double term1_a1 = 2.0 * std::sqrt(M_E) / (h_l[0] + std::sqrt(h_l[0] * h_l[0] + 4.0));
  // double term2_a1 = std::exp( 0.25 * (2.0 * h_l[0] - h_l[0] *
  //                   std::sqrt(h_l[0] * h_l[0] + 4.0)));
  double eq_a1 = h_l[0] + (2.0 * std::sqrt(M_E) / (h_l[0] + std::sqrt(h_l[0] * h_l[0] + 4.0))) *
      (std::exp( 0.25 * (2.0 * h_l[0] - h_l[0] * std::sqrt(h_l[0] * h_l[0] + 4.0))));
  bool a1 = (h_l[0] >= 0) && (h_u[0] > eq_a1);
  
  // Algorithm (2): Use -x ~ N_+ (-mu, -mu^+, sigma^2) on page 123. lower==-INFINITY
  // rejection sampling with exponential proposal. Use if upper < mean.
  // double term1_a2 = 2.0 * std::sqrt(M_E) / (-h_u[0] + std::sqrt(h_u[0]*h_u[0] + 4.0));
  // double term2_a2 = std::exp( 0.25 * (2.0 * h_u[0] + h_u[0] * std::sqrt(h_u[0]*h_u[0] + 4.0)));
  double eq_a2 = -h_u[0] + (2.0 * std::sqrt(M_E) / (-h_u[0] + std::sqrt(h_u[0]*h_u[0] + 4.0))) *
      (std::exp( 0.25 * (2.0 * h_u[0] + h_u[0] * std::sqrt(h_u[0]*h_u[0] + 4.0))));
  bool a2 = (h_u[0] <= 0) && (-h_l[0] > eq_a2);

  cudaMalloc((void**) &d_state, (*n) * sizeof(curandState_t));
  cudaMalloc((void**) &d_out,   (*n) * sizeof(double));
  cudaMalloc((void**) &d_mean,  (1)  * sizeof(double));
  cudaMalloc((void**) &d_sd,    (1)  * sizeof(double));
  cudaMalloc((void**) &d_l,     (1)  * sizeof(double));
  cudaMalloc((void**) &d_u,     (1)  * sizeof(double));
  
  cudaMemcpy(d_mean, h_mean, sizeof(double)*(1), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sd,   h_sd,   sizeof(double)*(1), cudaMemcpyHostToDevice);
  cudaMemcpy(d_l,    h_l,    sizeof(double)*(1), cudaMemcpyHostToDevice);
  cudaMemcpy(d_u,    h_u,    sizeof(double)*(1), cudaMemcpyHostToDevice);

  // Because I put 'curand_init(seed, i, 0, &state[i])' inside individual kernel,
  // no 'init<<<nblocks, nthreads>>>(time(0), d_states, (*n));'.
  if (a0) {
      // printf("Algorithm (0): Use Naive A-R method");
      rtnorm0<<<nblocks, nthreads>>>(time(0), d_state, *n, d_mean, d_sd, d_l, d_u, d_out);
  } else if (a1) {
      // printf("Algorithm (1): upper==INFINITY; left truncation");
      rtnorm1<<<nblocks, nthreads>>>(time(0), d_state, *n, d_mean, d_sd, d_l, d_u, d_out);
  } else if (a2) {
      // printf("Algorithm (2): lower==-INFINITY; right truncation");
      rtnorm2<<<nblocks, nthreads>>>(time(0), d_state, *n, d_mean, d_sd, d_l, d_u, d_out);
  } else {
      // rejection sampling with uniform proposal. Use if bounds are narrow and central.
      // printf("Algorithm (3): 2-sided truncation");
      rtnorm3<<<nblocks, nthreads>>>(time(0), d_state, *n, d_mean, d_sd, d_l, d_u, d_out);
  }

  cudaMemcpy(out, d_out, (*n) * sizeof(double), cudaMemcpyDeviceToHost);
  free(h_mean);
  free(h_sd);
  free(h_l);
  free(h_u);
  cudaFree(d_state);
  cudaFree(d_out);
  cudaFree(d_mean);
  cudaFree(d_sd);
  cudaFree(d_l);
  cudaFree(d_u);
}


void rlba_gpu(int *n, double *b,double *A, double *v, int *nv, double *sv,
              double *t0, int *nThreads, double *RT, int *R)
{
  // Allocate device memory 
  curandState_t *d_states;
  int *d_R, *d_n;
  double *d_RT, *d_b, *d_A, *d_v, *d_sv, *d_t0, *d_l, *d_u, *d_alphaStar;
  double *h_l, *h_u, *h_alphaStar;

  // Define the execution configuration
  int nThr = *nThreads;
  int nBlk = (*n)/(nThr) + 1;

  h_l  = (double *)malloc(nv[0] * sizeof(double));
  h_u  = (double *)malloc(nv[0] * sizeof(double));
  h_alphaStar = (double *)malloc(nv[0] * sizeof(double));

  for(int i=0; i<nv[0]; i++) {
      h_l[i]  = (0 - v[i]) / *sv; // convert to mean=0, sd=1
      h_u[i]  = (INFINITY - v[i]) / *sv; // Should also be infinity
      h_alphaStar[i] = 0.5 * (sqrt(h_l[i]*h_l[i] + 4.0) + h_l[i]); // use in rtnorm1_device, alphaStar  must be greater than 0

  }

  cudaMalloc((void**) &d_states, (*n) * sizeof(curandState_t));
  cudaMalloc((void**) &d_RT,     (*n) * sizeof(double));
  cudaMalloc((void**) &d_R,      (*n) * sizeof(int));
  cudaMalloc((void**) &d_n,       1  * sizeof(int));
  cudaMalloc((void**) &d_b,       1  * sizeof(double));
  cudaMalloc((void**) &d_A,       1  * sizeof(double));
  cudaMalloc((void**) &d_sv,      1  * sizeof(double));
  cudaMalloc((void**) &d_t0,      1  * sizeof(double));
  cudaMalloc((void**) &d_v,       nv[0]  * sizeof(double));
  cudaMalloc((void**) &d_l,       nv[0]  * sizeof(double));
  cudaMalloc((void**) &d_u,       nv[0]  * sizeof(double));
  cudaMalloc((void**) &d_alphaStar, nv[0] * sizeof(double));

  cudaMemcpy(d_n,  n,  1*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b,  b,  1*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_A,  A,  1*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sv, sv, 1*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_t0, t0, 1*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v,  v,   nv[0]*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_l,  h_l, nv[0]*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_u,  h_u, nv[0]*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_alphaStar, h_alphaStar,  nv[0]*sizeof(double), cudaMemcpyHostToDevice);

  rlba<<<nBlk, nThr>>>(d_states, d_n, d_b, d_A, d_v, d_sv, d_t0, d_l, d_u, d_alphaStar, d_RT, d_R);
  cudaMemcpy(RT, d_RT, (*n) * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(R,  d_R,  (*n) * sizeof(int), cudaMemcpyDeviceToHost);

   free(h_l);
   free(h_u);
   free(h_alphaStar);

   cudaFree(d_b);
   cudaFree(d_A);
   cudaFree(d_v);
   cudaFree(d_sv);
   cudaFree(d_t0);
   cudaFree(d_l);
   cudaFree(d_u);
   cudaFree(d_alphaStar);
 
   cudaFree(d_states);
   cudaFree(d_RT);
   cudaFree(d_R);
}



/*
void rlba_gpu(int *n, double *b,double *A, double* v1, double* v2, double* sv,
              double* t0, int *nThreads, double *RT, int *R)
{
  // Allocate device memory 
  curandState_t *d_states;
  int *d_R;
  double *d_RT, *d_b, *d_A, *d_v1, *d_v2, *d_sv, *d_t0, *d_l, *d_u, *d_alphaStar;
  double *h_b, *h_A, *h_v1, *h_v2, *h_sv, *h_t0, *h_l, *h_u, *h_alphaStar;

  // Define the execution configuration
  int nThr = *nThreads;
  int nBlk = (*n)/(nThr) + 1;

  h_b  = (double *)malloc(1 * sizeof(double));
  h_A  = (double *)malloc(1 * sizeof(double));
  h_v1 = (double *)malloc(1 * sizeof(double));
  h_v2 = (double *)malloc(1 * sizeof(double));
  h_sv = (double *)malloc(1 * sizeof(double));
  h_t0 = (double *)malloc(1 * sizeof(double));
  h_l  = (double *)malloc(2 * sizeof(double));
  h_u  = (double *)malloc(2 * sizeof(double));
  h_alphaStar = (double *)malloc(2 * sizeof(double));

  h_b[0]  = *b;
  h_A[0]  = *A;
  h_v1[0] = *v1;
  h_v2[0] = *v2;
  h_sv[0] = *sv;
  h_t0[0] = *t0;

  h_l[0]  = (0 - *v1) / *sv; // convert to mean=0, sd=1
  h_l[1]  = (0 - *v2) / *sv; // convert to mean=0, sd=1
  h_u[0]  = (INFINITY - *v1) / *sv; // Should also be infinity
  h_u[1]  = (INFINITY - *v2) / *sv; // Should also be infinity
  h_alphaStar[0] = 0.5 * (sqrt(h_l[0]*h_l[0] + 4.0) + h_l[0]); // use in rtnorm1_device, alphaStar  must be greater than 0
  h_alphaStar[1] = 0.5 * (sqrt(h_l[1]*h_l[1] + 4.0) + h_l[1]); 


  printf("host parameters %f %f %f %f %f %f\n", h_b[0], h_A[0], h_v1[0], h_v2[0],
         h_sv[0], h_t0[0]);
  printf("dim thread %d and dim block, %d\n", nThr, nBlk);
  

  cudaMalloc((void**) &d_states, (*n) * sizeof(curandState_t));
  cudaMalloc((void**) &d_RT,     (*n) * sizeof(double));
  cudaMalloc((void**) &d_R,      (*n) * sizeof(int));
  cudaMalloc((void**) &d_b,       1  * sizeof(double));
  cudaMalloc((void**) &d_A,       1  * sizeof(double));
  cudaMalloc((void**) &d_v1,      1  * sizeof(double));
  cudaMalloc((void**) &d_v2,      1  * sizeof(double));
  cudaMalloc((void**) &d_sv,      1  * sizeof(double));
  cudaMalloc((void**) &d_t0,      1  * sizeof(double));
  cudaMalloc((void**) &d_l,       2  * sizeof(double));
  cudaMalloc((void**) &d_u,       2  * sizeof(double));
  cudaMalloc((void**) &d_alphaStar, 2 * sizeof(double));

  cudaMemcpy(d_b,  h_b,  1*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_A,  h_A,  1*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v1, h_v1, 1*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v2, h_v2, 1*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sv, h_sv, 1*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_t0, h_t0, 1*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_l,  h_l,  2*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_u,  h_u,  2*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_alphaStar, h_alphaStar,  2*sizeof(double), cudaMemcpyHostToDevice);

  rlba<<<nBlk, nThr>>>(d_states, *n, d_b, d_A, d_v1, d_v2, d_sv, d_t0, d_l, d_u, d_alphaStar, d_RT, d_R);
  cudaMemcpy(RT, d_RT, (*n) * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(R,  d_R,  (*n) * sizeof(int), cudaMemcpyDeviceToHost);

  free(h_b);
  free(h_A);
  free(h_v1);
  free(h_v2);
  free(h_sv);
  free(h_t0);
  free(h_l);
  free(h_u);
  free(h_alphaStar);
  
  cudaFree(d_b);
  cudaFree(d_A);
  cudaFree(d_v1);
  cudaFree(d_v2);
  cudaFree(d_sv);
  cudaFree(d_t0);
  cudaFree(d_l);
  cudaFree(d_u);
  cudaFree(d_alphaStar);

  cudaFree(d_states);
  cudaFree(d_RT);
  cudaFree(d_R);
}

*/


/*
__global__ void randoms_lba(curandState_t* states, int n, double* b1, double* b2,
                            double* A1, double* A2, double* mu1, double* mu2,
                            double* sigma1, double* sigma2, double* t01, double* t02,
                            double* rts, int* responses) {
    
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < n; i += numThreads)
    {
        double x01 = A1[0] * curand_uniform_double(&states[i]);
        double x02 = A2[0] * curand_uniform_double(&states[i]);
        double v1  = mu1[0] + sigma1[0] * curand_normal_double(&states[i]);
        double v2  = mu2[0] + sigma2[0] * curand_normal_double(&states[i]);
        // if(v1 < 0) {v1 = 1e6;}         
        // if(v2 < 0) {v2 = 1e6;}         
        double dt1 = (b1[0] - x01) / v1;
        double dt2 = (b2[0] - x02) / v2;
        if(dt1 < dt2) {
            rts[i] = dt1 + t01[0];
            responses[i] = 1;
        } else {
            rts[i] = dt2 + t02[0];
            responses[i] = 2;
        }
    }
    
}
*/
