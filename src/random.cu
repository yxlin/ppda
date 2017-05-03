#include <unistd.h>
#include <stdio.h>  // C printing
#include <curand.h>        // Host random API
#include <curand_kernel.h> // Device random API
#include "../inst/include/common.h"  
#include "../inst/include/constant.h"  
#include "../inst/include/util.h"
#include "../inst/include/reduce.h"  
#include <armadillo> // Armadillo vector operations

extern "C" void runif_entry(int *n, double *min, double *max, int *nth, bool *dp,
                            double *out);
extern "C" void rnorm_entry(int *n, double *mean, double *sd, int *nth, bool *dp,
                            double *out);
extern "C" void rtnorm_entry(int *n, double *mean, double *sd, double *l,
                             double *u, int *nth, bool *dp, double *out);
extern "C" void rlba_entry_double(int *n, double *b, double *A, double *mean_v,
                                  int *nmean_v, double *sd_v, int *nsd_v,
                                  double *t0, int *nth, double *RT, int *R);
extern "C" void rlba_entry_float(int *n, double *b, double *A, double *mean_v,
                                  int *nmean_v, double *sd_v, int *nsd_v,
                                  double *t0, int *nth, double *RT, int *R);

extern "C" void rlba_n1_double(int *n, double *b, double *A, double *mean_v,
                               int *nmean_v, double *sd_v, int *nsd_v,
                               double *t0, int *nth, double *RT, int *R);
extern "C" void rlba_n1_float(int *n, double *b, double *A, double *mean_v,
                               int *nmean_v, double *sd_v, int *nsd_v,
                               double *t0, int *nth, double *RT, int *R);

__global__ void runif_kernel(int n, double* min, double* max, double* out) {
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
    // 'clock64()' allows each run produces different numbers.
    // '+ threadID' allows each thread generates different numbers.
    // Not place threadID on the 2nd argument speed up for 100 times.
    curandState_t state;
    curand_init((clock64() << 20)+threadID, 0, 0, &state);
    for (size_t i = threadID; i < n; i += numThreads) {
       out[i] = curand_uniform_double(&state);
       out[i] *= (max[0] - min[0] + 0.0000001);
       out[i] += min[0];
    }
}

__global__ void runif_kernel(int n, float* min, float* max, float* out) {
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
    curandState_t state;
    curand_init((clock64() << 20)+threadID, 0, 0, &state);
    for (size_t i = threadID; i < n; i += numThreads) {
       out[i] = curand_uniform(&state);
       out[i] *= (max[0] - min[0] + 0.0000001);
       out[i] += min[0];
    }
}

__global__ void rnorm_kernel(int n, double* mean, double* sd, double* out) {
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    curandState_t state;
    curand_init( (clock64() << 20) + threadID, 0, 0, &state);  
    for (size_t i = threadID; i < n; i += numThreads) {
        out[i] = mean[0] + sd[0]*curand_normal_double(&state);
    }
}

__global__ void rnorm_kernel(int n, float* mean, float* sd, float* out) {
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    curandState_t state;
    curand_init( (clock64() << 20) + threadID, 0, 0, &state);  
    for (size_t i = threadID; i < n; i += numThreads) {
        out[i] = mean[0] + sd[0]*curand_normal(&state);
    }
}

__global__ void rtnorm0_kernel(int n, double* mean, double* sd, double* l,
                        double* u, double* out) {
    // Accept-Reject Algorithm 0;
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    int valid;
    double z;
    curandState_t state;
    curand_init( (clock64() << 20) + threadID, 0, 0, &state);  
    for (size_t i = threadID; i < n; i += numThreads) {
        valid = 0;
        while (valid == 0) {
           z = curand_normal_double(&state);
           if (z <= *u && z >= *l) {
               out[i] = z*(*sd) + *mean;
               break;
           }
        }
    }
}

__global__ void rtnorm1_kernel(int n, double* mean, double* sd, double* l,
                        double* u, double* out) {
     // Algorithm 1; 'expl'; use when lower > mean; upper = INFINITY
     const int numThreads = blockDim.x * gridDim.x;
     const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
     int valid;              // alphaStar must be greater than 0
     double z, r, num; // a stands for alphaStar in Robert (1995)
     double a = 0.5 * (sqrt(l[0] * l[0] + 4.0) + l[0]);
     curandState_t state;
     curand_init( (clock64() << 20) + threadID, 0, 0, &state);  

     for (size_t i = threadID; i < n; i += numThreads) {
         valid = 0;
         while (valid == 0) {
           // rexp via runif. a=0, b=1/alphaStar; see Saucier (2000)
           //   assert( b > 0.);
           //   return a - b * log( uniform(0., 1.)) 
           z   = (-1/a) * log(curand_uniform_double(&state)) + l[0];
           num = curand_uniform_double(&state);
           r   = exp(-0.5 * (z - a) * (z - a));
           if (num <= r && z <= *u) {
             out[i] = z*(*sd) + *mean;
             break;
           }
         }
     }
}

__global__ void rtnorm2_kernel(int n, double* mean, double* sd, double* l,
                        double* u, double* out) {
     // Algorithm 2; 'expu'; use when upper < mean; lower = -INFINITY.
     const int numThreads = blockDim.x * gridDim.x;
     const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
     int valid;
     double z, r, num; // a stands for alphaStar in Robert (1995)
     double a = 0.5 * (sqrt(u[0] * u[0] + 4.0) - u[0]);
     curandState_t state;
     curand_init( (clock64() << 20) + threadID, 0, 0, &state);  

     for (size_t i = threadID; i < n; i += numThreads) {
         valid = 0;
         while (valid == 0) {
             z   = (-1/a) * log(curand_uniform_double(&state)) - u[0];
             num = curand_uniform_double(&state);
             r   = exp(-0.5 * (z - a) * (z - a));
             if (num <= r && z <= -l[0]) {
                out[i] = -z*(*sd) + *mean; // note '-' before z
                break;
             }
         }
     }
}

__global__ void rtnorm3_kernel(int n, double* mean, double* sd, double* l,
                        double* u, double* out) {
    // Algorithm 3; u;  page 123. 2.2. Two-sided truncated normal dist.
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
    int valid;
    double z, r, num;
    double maxminusmin = u[0] - l[0] + 1e-7;
    double lower2 = l[0] * l[0];
    double upper2 = u[0] * u[0];
    curandState_t state;
    curand_init( (clock64() << 20) + threadID, 0, 0, &state);  

    for (size_t i = threadID; i < n; i += numThreads) {
        valid = 0;
        while (valid == 0) {
            z = l[0] + maxminusmin*curand_uniform_double(&state);
            if (l[0] > 0) {
                r = exp( 0.5 * (lower2 - z*z) );
            } else if (u[0] < 0) {
                r = exp( 0.5 * (upper2 - z*z) );
            } else { 
                r = exp( -0.5 * z * z );
            }

            num = curand_uniform_double(&state);
            if (num <= r) {
              out[i] = z*(*sd) + *mean;
              break;
            }
        }
    }
}

__global__ void rtnorm0_kernel(int n, float* mean, float* sd, float* l,
                        float* u, float* out) {
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
    int valid;
    double z;
    curandState_t state;
    curand_init( (clock64() << 20) + threadID, 0, 0, &state);  
    for (size_t i = threadID; i < n; i += numThreads) {
        valid = 0;
        while (valid == 0) {
           z = curand_normal(&state);
           if (z <= *u && z >= *l) {
               out[i] = z*(*sd) + *mean;
               break;
           }
        }
    }
}

__global__ void rtnorm1_kernel(int n, float* mean, float* sd, float* l,
                        float* u, float* out) {
     // Algorithm 1; 'expl'; use when lower > mean; upper = INFINITY
     const int numThreads = blockDim.x * gridDim.x;
     const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
     int valid;              // alphaStar must be greater than 0
     float z, r, num; // a stands for alphaStar in Robert (1995)
     float a = 0.5 * (sqrt(l[0] * l[0] + 4.0) + l[0]);
     curandState_t state;
     curand_init( (clock64() << 20) + threadID, 0, 0, &state);  

     for (size_t i = threadID; i < n; i += numThreads) {
         valid = 0;
         while (valid == 0) {
           z   = (-1/a) * log(curand_uniform(&state)) + l[0];
           num = curand_uniform(&state);
           r   = exp(-0.5 * (z - a) * (z - a));
           if (num <= r && z <= *u) {
             out[i] = z*(*sd) + *mean;
             break;
           }
         }
     }
}

__global__ void rtnorm2_kernel(int n, float* mean, float* sd, float* l,
                        float* u, float* out) {
     const int numThreads = blockDim.x * gridDim.x;
     const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
     int valid;
     float z, r, num; // a stands for alphaStar in Robert (1995)
     float a = 0.5 * (sqrt(u[0] * u[0] + 4.0) - u[0]);
     curandState_t state;
     curand_init( (clock64() << 20) + threadID, 0, 0, &state);  

     for (size_t i = threadID; i < n; i += numThreads) {
         valid = 0;
         while (valid == 0) {
             z   = (-1/a) * log(curand_uniform(&state)) - u[0];
             num = curand_uniform(&state);
             r   = exp(-0.5 * (z - a) * (z - a));
             if (num <= r && z <= -l[0]) {
                out[i] = -z*(*sd) + *mean; // note '-' before z
                break;
             }
         }
     }
}

__global__ void rtnorm3_kernel(int n, float* mean, float* sd, float* l,
                        float* u, float* out) {
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
    int valid;
    float z, r, num;
    float maxminusmin = u[0] - l[0] + 1e-7;
    float lower2 = l[0] * l[0];
    float upper2 = u[0] * u[0];
    curandState_t state;
    curand_init( (clock64() << 20) + threadID, 0, 0, &state);  

    for (size_t i = threadID; i < n; i += numThreads) {
        valid = 0;
        while (valid == 0) {
            z = l[0] + maxminusmin*curand_uniform(&state);
            if (l[0] > 0) {
                r = exp( 0.5 * (lower2 - z*z) );
            } else if (u[0] < 0) {
                r = exp( 0.5 * (upper2 - z*z) );
            } else { 
                r = exp( -0.5 * z * z );
            }

            num = curand_uniform(&state);
            if (num <= r) {
              out[i] = z*(*sd) + *mean;
              break;
            }
        }
    }
}

static __device__ inline
double rtnorm0_device(curandState_t* state, double mean, double sd,
                      double l, double u) {
    double z, out; 
    unsigned int valid = 0;
    while (valid == 0) {
         z = curand_normal_double(&state[0]);
         if (z <= u && z >= l) {
              out = z*sd + mean;
              break;
         }
    }
   return out;
}

static __device__ inline
float rtnorm0_device(curandState_t* state, float mean, float sd,
                      float l, float u) {
    float z, out; 
    unsigned int valid = 0;
    while (valid == 0) {
         z = curand_normal(&state[0]);
         if (z <= u && z >= l) {
              out = z*sd + mean;
              break;
         }
    }
   return out;
}

static __device__ inline
double rtnorm1_device(curandState_t* state, double a, double mean,
                      double sd, double l, double u) {
    double z, r, num, out; // a stands for alphaStar in Robert (1995)
    unsigned int valid = 0;
    while (valid == 0) {
         z   = (-1/a) * log(curand_uniform_double(&state[0])) + l;
         num = curand_uniform_double(&state[0]);
         r   = exp(-0.5 * (z - a) * (z - a));
         if (num <= r && z <= u) {
              out = z*sd + mean;
              break;
         }
    }
   return out;
}

static __device__ inline
float rtnorm1_device(curandState_t* state, float a, float mean,
                      float sd, float l, float u) {
    float z, r, num, out; // a stands for alphaStar in Robert (1995)
    unsigned int valid = 0;
    while (valid == 0) {
         z   = (-1/a) * log(curand_uniform(&state[0])) + l;
         num = curand_uniform(&state[0]);
         r   = exp(-0.5 * (z - a) * (z - a));
         if (num <= r && z <= u) {
              out = z*sd + mean;
              break;
         }
    }
   return out;
}


__global__ void rlba_kernel(unsigned int* n, double* b, double* A, double* mean_v,
                            double* sd_v, double* t0, double* lower,
                            double* upper, double* a, bool* c, double* RT,
                            unsigned int* R)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
    double v0, v1, dt0, dt1;

    curandState_t state;
    curand_init( (clock64() << 20) + threadID, 0, 0, &state);
    for (size_t i = threadID; i < *n; i += numThreads)
    {
        v0 = c[0] ? rtnorm0_device(&state, mean_v[0], sd_v[0], lower[0], upper[0]) :
                    rtnorm1_device(&state, a[0], mean_v[0], sd_v[0], lower[0], upper[0]);
        v1 = c[1] ? rtnorm0_device(&state, mean_v[1], sd_v[1], lower[1], upper[1]) :
                    rtnorm1_device(&state, a[1], mean_v[1], sd_v[1], lower[1], upper[1]);
        dt0 = (b[0] - A[0] * curand_uniform_double(&state)) / v0;
        dt1 = (b[0] - A[0] * curand_uniform_double(&state)) / v1;
        RT[i] = (dt0 < dt1) ? (dt0 + t0[0]) : (dt1 + t0[0]);
        R[i]  = (dt0 < dt1) ? 1 : 2;
    }
}


__global__ void rlba_kernel(unsigned int* n, float* b, float* A, float* mean_v,
                            float* sd_v, float* t0, float* lower,
                            float* upper, float* a, bool* c, float* RT,
                            unsigned int* R)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
    float v0, v1, dt0, dt1;
    curandState_t state;
    curand_init( (clock64() << 20) + threadID, 0, 0, &state);
    for (size_t i = threadID; i < n[0]; i += numThreads)
    {
        v0 = c[0] ? rtnorm0_device(&state, mean_v[0], sd_v[0], lower[0], upper[0]) :
                    rtnorm1_device(&state, a[0], mean_v[0], sd_v[0], lower[0], upper[0]);
        v1 = c[1] ? rtnorm0_device(&state, mean_v[1], sd_v[1], lower[1], upper[1]) :
                    rtnorm1_device(&state, a[1], mean_v[1], sd_v[1], lower[1], upper[1]);
        dt0 = (b[0] - A[0] * curand_uniform(&state)) / v0;
        dt1 = (b[0] - A[0] * curand_uniform(&state)) / v1;
        RT[i] = (dt0 < dt1) ? (dt0 + t0[0]) : (dt1 + t0[0]);
        R[i]  = (dt0 < dt1) ? 1 : 2;
    }
}

__global__ void rlba_n1_kernel(unsigned int *n, double *b, double *A,
                               double *mean_v, double *sd_v, double *t0,
                               double *lower, double *upper, double *a, bool *c,
                               double *RT0, unsigned int *R) {
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
    double v0, v1, dt0, dt1;

    curandState_t state;
    curand_init( (clock64() << 20) + threadID, 0, 0, &state);
    for (size_t i = threadID; i < *n; i += numThreads)
    {
        v0 = c[0] ? rtnorm0_device(&state, mean_v[0], sd_v[0], lower[0], upper[0]) :
                    rtnorm1_device(&state, a[0], mean_v[0], sd_v[0], lower[0], upper[0]);
        v1 = c[1] ? rtnorm0_device(&state, mean_v[1], sd_v[1], lower[1], upper[1]) :
                    rtnorm1_device(&state, a[1], mean_v[1], sd_v[1], lower[1], upper[1]);
        dt0 = (b[0] - A[0] * curand_uniform_double(&state)) / v0;
        dt1 = (b[0] - A[0] * curand_uniform_double(&state)) / v1;
        RT0[i] = dt0 < dt1 ? dt0 + t0[0] : 0;
        R[i]   = dt0 < dt1 ? 1 : 2;
    }
}

__global__ void rlba_n1_kernel(unsigned int *n, float *b, float *A, float *mean_v,
                               float *sd_v, float *t0, float *lower, float *upper,
                               float *a, bool *c, float *RT0, unsigned int *R) {
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
    float v0, v1, dt0, dt1;

    curandState_t state;
    curand_init( (clock64() << 20) + threadID, 0, 0, &state);
    for (unsigned int i = threadID; i < *n; i += numThreads)
    {
        v0 = c[0] ? rtnorm0_device(&state, mean_v[0], sd_v[0], lower[0], upper[0]) :
                    rtnorm1_device(&state, a[0], mean_v[0], sd_v[0], lower[0], upper[0]);
        v1 = c[1] ? rtnorm0_device(&state, mean_v[1], sd_v[1], lower[1], upper[1]) :
                    rtnorm1_device(&state, a[1], mean_v[1], sd_v[1], lower[1], upper[1]);
        dt0 = (b[0] - A[0] * curand_uniform(&state)) / v0;
        dt1 = (b[0] - A[0] * curand_uniform(&state)) / v1;
        RT0[i] = dt0 < dt1 ? dt0 + t0[0] : 0;
        R[i]   = dt0 < dt1 ? 1 : 2;
    }
}

void runif_entry(int *n, double *min, double *max, int *nth, bool *dp, double *out)
{
    if(*dp) {
        size_t ndSize = *n * sizeof(double);
        size_t dSize  = 1 * sizeof(double);
        double *d_min, *d_max, *d_out, *h_out;
        CHECK(cudaMalloc((void**) &d_out,   ndSize));
        CHECK(cudaMalloc((void**) &d_min,   dSize));
        CHECK(cudaMalloc((void**) &d_max,   dSize));
        CHECK(cudaHostAlloc((void**)&h_out, ndSize, cudaHostAllocDefault));
        CHECK(cudaMemcpy(d_min, min, dSize, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_max, max, dSize, cudaMemcpyHostToDevice));
        runif_kernel<<<((*n)/(*nth) + 1), *nth>>>((*n), d_min, d_max, d_out);
        CHECK(cudaMemcpy(h_out, d_out, ndSize, cudaMemcpyDeviceToHost));
        for(size_t i=0; i<*n; i++) { out[i] = h_out[i]; }
        cudaFreeHost(h_out);   cudaFree(d_out);
        cudaFree(d_min);       cudaFree(d_max);
    } else {
        size_t nfSize = *n * sizeof(float);
        size_t fSize  = 1 * sizeof(float);
        float *d_min, *d_max, *d_out;
        float *h_min, *h_max, *h_out;
        h_min = (float *)malloc(fSize);
        h_max = (float *)malloc(fSize);
        *h_min = (float)*min;
        *h_max = (float)*max;

        CHECK(cudaMalloc((void**) &d_out,   nfSize));
        CHECK(cudaMalloc((void**) &d_min,   fSize));
        CHECK(cudaMalloc((void**) &d_max,   fSize));
        CHECK(cudaHostAlloc((void**)&h_out, nfSize, cudaHostAllocDefault));
        CHECK(cudaMemcpy(d_min, h_min, fSize, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_max, h_max, fSize, cudaMemcpyHostToDevice));
        runif_kernel<<<((*n)/(*nth) + 1), *nth>>>((*n), d_min, d_max, d_out);
        CHECK(cudaMemcpy(h_out, d_out, nfSize, cudaMemcpyDeviceToHost));
        for(size_t i=0; i<*n; i++) { out[i] = (double)h_out[i]; }
        free(h_min); free(h_max);
        cudaFreeHost(h_out);   cudaFree(d_out);
        cudaFree(d_min);       cudaFree(d_max);
    }
}



void rnorm_entry(int *n, double *mean, double *sd, int *nth, bool *dp, double *out)
{
    if (*dp) {
        double *d_out, *d_mean, *d_sd, *h_out;
        CHECK(cudaHostAlloc((void**)&h_out, *n * sizeof(double), cudaHostAllocDefault));
        CHECK(cudaMalloc((void**)&d_out,    *n * sizeof(double)));
        CHECK(cudaMalloc((void**)&d_mean,    1 * sizeof(double)));
        CHECK(cudaMalloc((void**)&d_sd,      1 * sizeof(double)));
        CHECK(cudaMemcpy(d_mean, mean,       1 * sizeof(double), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_sd, sd,           1 * sizeof(double), cudaMemcpyHostToDevice));
        rnorm_kernel<<<((*n)/(*nth) + 1), *nth>>>(*n, d_mean, d_sd, d_out);
        CHECK(cudaMemcpy(h_out, d_out, *n * sizeof(double), cudaMemcpyDeviceToHost));
        for(int i=0; i<*n; i++) { out[i] = h_out[i]; }
        cudaFreeHost(h_out); cudaFree(d_out);
        cudaFree(d_mean);    cudaFree(d_sd);
    } else {
        size_t nfSize = *n * sizeof(float);
        size_t fSize  = 1 * sizeof(float);
        float *h_mean, *h_sd, *h_out;
        float *d_mean, *d_sd, *d_out;
        h_mean  = (float *)malloc(fSize);
        h_sd    = (float *)malloc(fSize);
        *h_mean = (float)*mean;
        *h_sd   = (float)*sd;

        CHECK(cudaHostAlloc((void**)&h_out, nfSize, cudaHostAllocDefault));
        CHECK(cudaMalloc((void**)&d_out,  nfSize));
        CHECK(cudaMalloc((void**)&d_mean, fSize));
        CHECK(cudaMalloc((void**)&d_sd,   fSize));
        CHECK(cudaMemcpy(d_mean, h_mean,  fSize, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_sd,   h_sd,    fSize, cudaMemcpyHostToDevice));
        rnorm_kernel<<<((*n)/(*nth) + 1), *nth>>>(*n, d_mean, d_sd, d_out);
        CHECK(cudaMemcpy(h_out, d_out, nfSize, cudaMemcpyDeviceToHost));
        for(int i=0; i<*n; i++) { out[i] = (double)h_out[i]; }
        free(h_mean); free(h_sd);
        cudaFreeHost(h_out); cudaFree(d_out);
        cudaFree(d_mean);    cudaFree(d_sd);
    }
}


void rtnorm_entry(int *n, double *mean, double *sd, double *l, double *u,
                  int *nth, bool *dp, double *out)
{
    // Stage 1 --------------------------------------------------------------80
    double *h_stdu, *h_stdl;
    size_t fSize = 1 * sizeof(float);
    size_t dSize = 1 * sizeof(double);
    h_stdl = (double *)malloc(dSize);
    h_stdu = (double *)malloc(dSize);
    h_stdl[0] = (*l - *mean) / *sd; // convert to mean=0, sd=1
    h_stdu[0] = (*u - *mean) / *sd;

    // Algorithm (a0): Use Naive A-R method
    // Algorithm (a1): Use Proposition 2.3 with only lower truncation. upper==INFINITY
    // rejection sampling with exponential proposal. Use if lower > mean
    // Algorithm (a2): Use -x ~ N_+ (-mu, -mu^+, sigma^2) on page 123. lower==-INFINITY
    // rejection sampling with exponential proposal. Use if upper < mean.
    // Algorithm (a3, else): rejection sampling with uniform proposal. Use if bounds
    // are narrow and central.
    bool a0 = (h_stdl[0] < 0 && h_stdu[0]==INFINITY)  ||
            (h_stdl[0] == -INFINITY && h_stdu[0] > 0) ||
        (std::isfinite(h_stdl[0]) && std::isfinite(h_stdu[0]) && h_stdl[0] < 0 && h_stdu[0] > 0 &&
        (h_stdu[0] - h_stdl[0]) > SQRT_2PI);
    double eq_a1 = h_stdl[0] + (2.0 * std::sqrt(M_E) / (h_stdl[0] + std::sqrt(h_stdl[0] * h_stdl[0] + 4.0))) *
        (std::exp( 0.25 * (2.0 * h_stdl[0] - h_stdl[0] * std::sqrt(h_stdl[0] * h_stdl[0] + 4.0))));
    bool a1 = (h_stdl[0] >= 0) && (h_stdu[0] > eq_a1);
    double eq_a2 = -h_stdu[0] + (2.0 * std::sqrt(M_E) / (-h_stdu[0] + std::sqrt(h_stdu[0]*h_stdu[0] + 4.0))) *
        (std::exp( 0.25 * (2.0 * h_stdu[0] + h_stdu[0] * std::sqrt(h_stdu[0]*h_stdu[0] + 4.0))));
    bool a2 = (h_stdu[0] <= 0) && (-h_stdl[0] > eq_a2);

    // Stage 2 --------------------------------------------------------------80
    if (*dp) {
        size_t ndSize = *n * sizeof(double);
        double *h_out;
        double *d_out, *d_mean, *d_sd, *d_l, *d_u;
        CHECK(cudaHostAlloc((void**)&h_out, ndSize, cudaHostAllocDefault));
        CHECK(cudaMalloc((void**) &d_out,   ndSize));
        CHECK(cudaMalloc((void**) &d_mean,  dSize));
        CHECK(cudaMalloc((void**) &d_sd,    dSize));
        CHECK(cudaMalloc((void**) &d_l,     dSize));
        CHECK(cudaMalloc((void**) &d_u,     dSize));
        CHECK(cudaMemcpy(d_mean, mean, dSize, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_sd,     sd, dSize, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_l,  h_stdl, dSize, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_u,  h_stdu, dSize, cudaMemcpyHostToDevice));

        if (a0) {
            rtnorm0_kernel<<<((*n)/(*nth) + 1), *nth>>>(*n, d_mean, d_sd, d_l, d_u, d_out);
        } else if (a1) {
            rtnorm1_kernel<<<((*n)/(*nth) + 1), *nth>>>(*n, d_mean, d_sd, d_l, d_u, d_out);
        } else if (a2) {
            rtnorm2_kernel<<<((*n)/(*nth) + 1), *nth>>>(*n, d_mean, d_sd, d_l, d_u, d_out);
        } else {
            rtnorm3_kernel<<<((*n)/(*nth) + 1), *nth>>>(*n, d_mean, d_sd, d_l, d_u, d_out);
        }

        cudaMemcpy(h_out, d_out, ndSize, cudaMemcpyDeviceToHost);
        for(size_t i=0; i<*n; i++) { out[i] = h_out[i]; }
        free(h_stdl);  free(h_stdu);
        cudaFreeHost(h_out); cudaFree(d_out);
        cudaFree(d_mean);    cudaFree(d_sd);
        cudaFree(d_l);       cudaFree(d_u);
    } else {
        size_t nfSize = *n * sizeof(float);
        float *h_out, *h_mean, *h_sd, *h_l, *h_u;
        float *d_out, *d_mean, *d_sd, *d_l, *d_u;
        h_mean  = (float *)malloc(fSize);
        h_sd    = (float *)malloc(fSize);
        h_l     = (float *)malloc(fSize);
        h_u     = (float *)malloc(fSize);
        *h_mean = (float)*mean;
        *h_sd   = (float)*sd;
        *h_l    = (float)*h_stdl;
        *h_u    = (float)*h_stdu;

        CHECK(cudaHostAlloc((void**)&h_out, nfSize, cudaHostAllocDefault));
        CHECK(cudaMalloc((void**) &d_out,   nfSize));
        CHECK(cudaMalloc((void**) &d_mean,  fSize));
        CHECK(cudaMalloc((void**) &d_sd,    fSize));
        CHECK(cudaMalloc((void**) &d_l,     fSize));
        CHECK(cudaMalloc((void**) &d_u,     fSize));
        CHECK(cudaMemcpy(d_mean, h_mean, fSize, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_sd,   h_sd,   fSize, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_l,  h_l,      fSize, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_u,  h_u,      fSize, cudaMemcpyHostToDevice));

        if (a0) {
            rtnorm0_kernel<<<((*n)/(*nth) + 1), *nth>>>(*n, d_mean, d_sd, d_l, d_u, d_out);
        } else if (a1) {
            rtnorm1_kernel<<<((*n)/(*nth) + 1), *nth>>>(*n, d_mean, d_sd, d_l, d_u, d_out);
        } else if (a2) {
            rtnorm2_kernel<<<((*n)/(*nth) + 1), *nth>>>(*n, d_mean, d_sd, d_l, d_u, d_out);
        } else {
            rtnorm3_kernel<<<((*n)/(*nth) + 1), *nth>>>(*n, d_mean, d_sd, d_l, d_u, d_out);
        }

        cudaMemcpy(h_out, d_out, nfSize, cudaMemcpyDeviceToHost);
        for(size_t i=0; i<*n; i++) { out[i] = (double)h_out[i]; }
        free(h_stdl);  free(h_stdu); free(h_l); free(h_u); free(h_mean); free(h_sd);
        cudaFreeHost(h_out); cudaFree(d_out);
        cudaFree(d_mean);    cudaFree(d_sd);
        cudaFree(d_l);       cudaFree(d_u);
    }
}


void rlba_entry_double(int *n, double *b,double *A, double *mean_v, int *nmean_v,
                double *sd_v, int *nsd_v, double *t0, int *nth, double *RT,
                int *R)
{
    bool *h_c, *d_c; // c for choice switch for rtnorm insider rlba_kernel
    unsigned int *d_R, *d_n;
    unsigned int *h_R, *h_n;
    double *d_l, *d_u, *d_a, *d_RT;
    double *h_l, *h_u, *h_a, *h_RT; // *h_a and *d_a stands for alphaStart
    double *d_b, *d_A, *d_mean_v, *d_sd_v, *d_t0;

    size_t dSize  = 1 * sizeof(double);
    size_t vdSize = *nmean_v * dSize;
    size_t vbSize = *nmean_v * sizeof(bool);
    
    h_c = (bool *)malloc(vbSize);
    h_l = (double *)malloc(vdSize);
    h_u = (double *)malloc(vdSize);
    h_a = (double *)malloc(vdSize);

    for(int i=0; i<nmean_v[0]; i++) {
        h_l[i] = (0 - mean_v[i]) / sd_v[i];        // convert to mean=0, sd=1
        h_u[i] = (INFINITY - mean_v[i]) / sd_v[i]; // Should also be infinity
        h_a[i] = 0.5 * (sqrt(h_l[i]*h_l[i] + 4.0) + h_l[i]); // use in rtnorm1_device, alphaStar must be greater than 0
        h_c[i] = (h_l[i] < 0 && h_u[i]==INFINITY) || (h_l[i]==-INFINITY && h_u[i]) ||
            (isfinite(h_l[i]) && isfinite(h_u[i]) && h_l[i] < 0 && h_u[i] > 0 && ((h_u[i] - h_l[i]) > SQRT_2PI));
    }

    size_t uSize = 1 * sizeof(unsigned int);
    size_t ndSize = *n * dSize;
    size_t nuSize = *n * uSize;

    h_n = (unsigned int *)malloc(uSize);
    *h_n = (unsigned int)*n;
    
    CHECK(cudaHostAlloc((void**)&h_RT,   ndSize, cudaHostAllocDefault));
    CHECK(cudaHostAlloc((void**)&h_R,    nuSize, cudaHostAllocDefault));
    CHECK(cudaMalloc((void**) &d_R,      nuSize));
    CHECK(cudaMalloc((void**) &d_RT,     ndSize));
    CHECK(cudaMalloc((void**) &d_n,      uSize));
    CHECK(cudaMalloc((void**) &d_b,      dSize));
    CHECK(cudaMalloc((void**) &d_A,      dSize));
    CHECK(cudaMalloc((void**) &d_t0,     dSize));
    CHECK(cudaMalloc((void**) &d_sd_v,   vdSize));
    CHECK(cudaMalloc((void**) &d_mean_v, vdSize));
    CHECK(cudaMalloc((void**) &d_l,      vdSize));
    CHECK(cudaMalloc((void**) &d_u,      vdSize));
    CHECK(cudaMalloc((void**) &d_a,      vdSize));
    CHECK(cudaMalloc((void**) &d_c,      vbSize));

    CHECK(cudaMemcpy(d_n, h_n,  uSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b,   b,  dSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_A,   A,  dSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_t0, t0,  dSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_mean_v, mean_v, vdSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_sd_v,     sd_v, vdSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_l,         h_l, vdSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_u,         h_u, vdSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_a,         h_a, vdSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_c,         h_c, vbSize, cudaMemcpyHostToDevice));

    rlba_kernel<<<((*n)/(*nth) + 1), *nth>>>(d_n, d_b, d_A, d_mean_v, d_sd_v,
                                             d_t0, d_l, d_u, d_a, d_c, d_RT,
                                             d_R);
    CHECK(cudaMemcpy(h_RT, d_RT, ndSize, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_R,  d_R,  nuSize, cudaMemcpyDeviceToHost));
    for(size_t i=0; i<*n; i++) {
            RT[i] = h_RT[i];
            R[i]  = h_R[i];
    }

    cudaFreeHost(h_RT); cudaFreeHost(h_R); 
    cudaFree(d_RT);     cudaFree(d_R);    
    cudaFree(d_b);      cudaFree(d_A);    cudaFree(d_t0);
    cudaFree(d_mean_v); cudaFree(d_sd_v);
    cudaFree(d_l); cudaFree(d_u); cudaFree(d_a); cudaFree(d_c); cudaFree(d_n);
    free(h_c); free(h_l); free(h_u); free(h_a); free(h_n);
}

void rlba_entry_float(int *n, double *b,double *A, double *mean_v, int *nmean_v,
                double *sd_v, int *nsd_v, double *t0, int *nth, double *RT,
                int *R)
{
    bool *h_c, *d_c; // c for choice switch for rtnorm insider rlba_kernel
    unsigned int *d_R, *d_n;
    unsigned int *h_R, *h_n;
    float *d_l, *d_u, *d_a, *d_RT;
    float *h_l, *h_u, *h_a, *h_RT; // *h_a and *d_a stands for alphaStart
    float *d_b, *d_A, *d_mean_v, *d_sd_v, *d_t0;
    float *h_b, *h_A, *h_mean_v, *h_sd_v, *h_t0;

    size_t fSize  = 1 * sizeof(float);
    size_t vfSize = *nmean_v * fSize;
    size_t vbSize = *nmean_v * sizeof(bool);
    
    h_c = (bool *)malloc(vbSize);
    h_l = (float *)malloc(vfSize);
    h_u = (float *)malloc(vfSize);
    h_a = (float *)malloc(vfSize);
    h_mean_v = (float *)malloc(vfSize);
    h_sd_v   = (float *)malloc(vfSize);

    for(int i=0; i<*nmean_v; i++) {
        h_l[i] = (0 - mean_v[i]) / sd_v[i];        // convert to mean=0, sd=1
        h_u[i] = (INFINITY - mean_v[i]) / sd_v[i]; // Should also be infinity
        h_a[i] = 0.5 * (sqrt(h_l[i]*h_l[i] + 4.0) + h_l[i]); // use in rtnorm1_device, alphaStar must be greater than 0
        h_c[i] = (h_l[i] < 0 && h_u[i]==INFINITY) || (h_l[i]==-INFINITY && h_u[i]) ||
            (isfinite(h_l[i]) && isfinite(h_u[i]) && h_l[i] < 0 && h_u[i] > 0 && ((h_u[i] - h_l[i]) > SQRT_2PI));
        h_mean_v[i] = (float)mean_v[i];
        h_sd_v[i]   = (float)sd_v[i];
    }

    size_t uSize = 1 * sizeof(unsigned int);
    size_t nfSize = *n * fSize;
    size_t nuSize = *n * uSize;

    h_n  = (unsigned int *)malloc(uSize);
    h_b  = (float *)malloc(fSize);
    h_A  = (float *)malloc(fSize);
    h_t0 = (float *)malloc(fSize);
    *h_n  = (unsigned int)*n;
    *h_b  = (float)*b;
    *h_A  = (float)*A;
    *h_t0 = (float)*t0;
    
    CHECK(cudaHostAlloc((void**)&h_RT,   nfSize, cudaHostAllocDefault));
    CHECK(cudaHostAlloc((void**)&h_R,    nuSize, cudaHostAllocDefault));
    CHECK(cudaMalloc((void**) &d_R,      nuSize));
    CHECK(cudaMalloc((void**) &d_RT,     nfSize));
    CHECK(cudaMalloc((void**) &d_n,      uSize));
    CHECK(cudaMalloc((void**) &d_b,      fSize));
    CHECK(cudaMalloc((void**) &d_A,      fSize));
    CHECK(cudaMalloc((void**) &d_t0,     fSize));
    CHECK(cudaMalloc((void**) &d_sd_v,   vfSize));
    CHECK(cudaMalloc((void**) &d_mean_v, vfSize));
    CHECK(cudaMalloc((void**) &d_l,      vfSize));
    CHECK(cudaMalloc((void**) &d_u,      vfSize));
    CHECK(cudaMalloc((void**) &d_a,      vfSize));
    CHECK(cudaMalloc((void**) &d_c,      vbSize));

    CHECK(cudaMemcpy(d_n, h_n,  uSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b,  fSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_A, h_A,  fSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_t0,h_t0, fSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_mean_v, h_mean_v, vfSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_sd_v,   h_sd_v,   vfSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_l,         h_l,   vfSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_u,         h_u,   vfSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_a,         h_a,   vfSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_c,         h_c,   vbSize, cudaMemcpyHostToDevice));

    rlba_kernel<<<((*n)/(*nth) + 1), *nth>>>(d_n, d_b, d_A, d_mean_v, d_sd_v,
                                             d_t0, d_l, d_u, d_a, d_c, d_RT,
                                             d_R);
    CHECK(cudaMemcpy(h_RT, d_RT, nfSize, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_R,  d_R,  nuSize, cudaMemcpyDeviceToHost));
    for(size_t i=0; i<*n; i++) {
      RT[i] = h_RT[i];
      R[i]  = h_R[i];
    }

    cudaFreeHost(h_RT); cudaFreeHost(h_R);
    cudaFree(d_RT);     cudaFree(d_R);    
    cudaFree(d_b);      cudaFree(d_A);    cudaFree(d_t0);
    cudaFree(d_mean_v); cudaFree(d_sd_v);
    cudaFree(d_l); cudaFree(d_u); cudaFree(d_a); cudaFree(d_c); cudaFree(d_n);
    free(h_c); free(h_l); free(h_u); free(h_a); free(h_n);
    free(h_b);      free(h_A);    free(h_t0);
    free(h_mean_v); free(h_sd_v);
}

void rlba_n1_double(int *n, double *b,double *A, double *mean_v, int *nmean_v,
                    double *sd_v, int *nsd_v, double *t0, int *nth, double *RT, int *R)
{
    bool *h_c, *d_c; // c for choice switch for rtnorm insider rlba_kernel
    unsigned int *d_n, *d_R;
    unsigned int *h_n, *h_R;
    double *d_l, *d_u, *d_a, *d_RT;
    double *h_l, *h_u, *h_a, *h_RT; // *h_a and *d_a stands for alphaStart
    double *d_b, *d_A, *d_mean_v, *d_sd_v, *d_t0;

    size_t dSize  = 1 * sizeof(double);
    size_t vdSize = *nmean_v * dSize;
    size_t vbSize = *nmean_v * sizeof(bool);
    
    h_c = (bool *)malloc(vbSize);
    h_l = (double *)malloc(vdSize);
    h_u = (double *)malloc(vdSize);
    h_a = (double *)malloc(vdSize);

    for(int i=0; i<nmean_v[0]; i++) {
        h_l[i] = (0 - mean_v[i]) / sd_v[i];        // convert to mean=0, sd=1
        h_u[i] = (INFINITY - mean_v[i]) / sd_v[i]; // Should also be infinity
        h_a[i] = 0.5 * (sqrt(h_l[i]*h_l[i] + 4.0) + h_l[i]); // use in rtnorm1_device, alphaStar must be greater than 0
        h_c[i] = (h_l[i] < 0 && h_u[i]==INFINITY) || (h_l[i]==-INFINITY && h_u[i]) ||
            (isfinite(h_l[i]) && isfinite(h_u[i]) && h_l[i] < 0 && h_u[i] > 0 && ((h_u[i] - h_l[i]) > SQRT_2PI));
    }

    size_t uSize = 1 * sizeof(unsigned int);
    size_t ndSize = *n * dSize;
    size_t nuSize = *n * uSize;

    h_n = (unsigned int *)malloc(uSize);
    *h_n = (unsigned int)*n;
    
    CHECK(cudaHostAlloc((void**)&h_RT,   ndSize, cudaHostAllocDefault));
    CHECK(cudaHostAlloc((void**)&h_R,    nuSize, cudaHostAllocDefault));
    CHECK(cudaMalloc((void**) &d_RT,     ndSize));
    CHECK(cudaMalloc((void**) &d_R,      nuSize));
    CHECK(cudaMalloc((void**) &d_n,      uSize));
    CHECK(cudaMalloc((void**) &d_b,      dSize));
    CHECK(cudaMalloc((void**) &d_A,      dSize));
    CHECK(cudaMalloc((void**) &d_t0,     dSize));
    CHECK(cudaMalloc((void**) &d_sd_v,   vdSize));
    CHECK(cudaMalloc((void**) &d_mean_v, vdSize));
    CHECK(cudaMalloc((void**) &d_l,      vdSize));
    CHECK(cudaMalloc((void**) &d_u,      vdSize));
    CHECK(cudaMalloc((void**) &d_a,      vdSize));
    CHECK(cudaMalloc((void**) &d_c,      vbSize));

    CHECK(cudaMemcpy(d_n, h_n,  uSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b,   b,  dSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_A,   A,  dSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_t0, t0,  dSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_mean_v, mean_v, vdSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_sd_v,     sd_v, vdSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_l,         h_l, vdSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_u,         h_u, vdSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_a,         h_a, vdSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_c,         h_c, vbSize, cudaMemcpyHostToDevice));

    rlba_n1_kernel<<<((*n)/(*nth) + 1), *nth>>>(d_n, d_b, d_A, d_mean_v, d_sd_v,
                                                d_t0, d_l, d_u, d_a, d_c, d_RT, d_R);
    CHECK(cudaMemcpy(h_RT, d_RT, ndSize, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_R,  d_R,  nuSize, cudaMemcpyDeviceToHost));
    for(size_t i=0; i<*n; i++) {
      RT[i] = h_RT[i];
      R[i]  = h_R[i];
    }

    cudaFreeHost(h_RT);  cudaFreeHost(h_R);
    cudaFree(d_RT);     cudaFree(d_R);
    cudaFree(d_b);      cudaFree(d_A);    cudaFree(d_t0);
    cudaFree(d_mean_v); cudaFree(d_sd_v);
    cudaFree(d_l); cudaFree(d_u); cudaFree(d_a); cudaFree(d_c); cudaFree(d_n);
    free(h_c); free(h_l); free(h_u); free(h_a); free(h_n);
}

void rlba_n1_float(int *n, double *b, double *A, double *mean_v, int *nmean_v,
                   double *sd_v, int *nsd_v, double *t0, int *nth, double *RT,
                   int *R)
{
    bool *h_c, *d_c; // c for choice switch for rtnorm insider rlba_kernel
    unsigned int *d_n, *d_R;
    unsigned int *h_n, *h_R;
    float *d_l, *d_u, *d_a, *d_RT;
    float *h_l, *h_u, *h_a, *h_RT; // *h_a and *d_a stands for alphaStart
    float *d_b, *d_A, *d_mean_v, *d_sd_v, *d_t0;
    float *h_b, *h_A, *h_mean_v, *h_sd_v, *h_t0;

    size_t fSize  = 1 * sizeof(float);
    size_t vfSize = *nmean_v * fSize;
    size_t vbSize = *nmean_v * sizeof(bool);
    
    h_c = (bool *)malloc(vbSize);
    h_l = (float *)malloc(vfSize);
    h_u = (float *)malloc(vfSize);
    h_a = (float *)malloc(vfSize);
    h_mean_v = (float *)malloc(vfSize);
    h_sd_v   = (float *)malloc(vfSize);

    for(int i=0; i<nmean_v[0]; i++) {
        h_l[i] = (0 - mean_v[i]) / sd_v[i];        // convert to mean=0, sd=1
        h_u[i] = (INFINITY - mean_v[i]) / sd_v[i]; // Should also be infinity
        h_a[i] = 0.5 * (sqrt(h_l[i]*h_l[i] + 4.0) + h_l[i]); // use in rtnorm1_device, alphaStar must be greater than 0
        h_c[i] = (h_l[i] < 0 && h_u[i]==INFINITY) || (h_l[i]==-INFINITY && h_u[i]) ||
            (isfinite(h_l[i]) && isfinite(h_u[i]) && h_l[i] < 0 && h_u[i] > 0 && ((h_u[i] - h_l[i]) > SQRT_2PI));
        h_mean_v[i] = (float)mean_v[i];
        h_sd_v[i]   = (float)sd_v[i];
    }

    size_t uSize = 1 * sizeof(unsigned int);
    size_t nfSize = *n * fSize;
    size_t nuSize = *n * uSize;

    h_n = (unsigned int *)malloc(uSize);
    h_b  = (float *)malloc(fSize);
    h_A  = (float *)malloc(fSize);
    h_t0 = (float *)malloc(fSize);
    *h_n = (unsigned int)*n;
    *h_b  = (float)*b;
    *h_A  = (float)*A;
    *h_t0 = (float)*t0;
    
    CHECK(cudaHostAlloc((void**)&h_RT,   nfSize, cudaHostAllocDefault));
    CHECK(cudaHostAlloc((void**)&h_R,    nuSize, cudaHostAllocDefault));
    CHECK(cudaMalloc((void**) &d_RT,     nfSize));
    CHECK(cudaMalloc((void**) &d_R,      nuSize));
    CHECK(cudaMalloc((void**) &d_n,      uSize));
    CHECK(cudaMalloc((void**) &d_b,      fSize));
    CHECK(cudaMalloc((void**) &d_A,      fSize));
    CHECK(cudaMalloc((void**) &d_t0,     fSize));
    CHECK(cudaMalloc((void**) &d_sd_v,   vfSize));
    CHECK(cudaMalloc((void**) &d_mean_v, vfSize));
    CHECK(cudaMalloc((void**) &d_l,      vfSize));
    CHECK(cudaMalloc((void**) &d_u,      vfSize));
    CHECK(cudaMalloc((void**) &d_a,      vfSize));
    CHECK(cudaMalloc((void**) &d_c,      vbSize));

    CHECK(cudaMemcpy(d_n, h_n,  uSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b,  fSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_A, h_A,  fSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_t0,h_t0, fSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_mean_v,h_mean_v,vfSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_sd_v,    h_sd_v,vfSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_l,         h_l, vfSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_u,         h_u, vfSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_a,         h_a, vfSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_c,         h_c, vbSize, cudaMemcpyHostToDevice));

    rlba_n1_kernel<<<((*n)/(*nth) + 1), *nth>>>(d_n, d_b, d_A, d_mean_v, d_sd_v,
                                                d_t0, d_l, d_u, d_a, d_c, d_RT,
                                                d_R);
    CHECK(cudaMemcpy(h_RT, d_RT, nfSize, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_R,  d_R,  nuSize, cudaMemcpyDeviceToHost));
    for(size_t i=0; i<*n; i++) {
      RT[i] = h_RT[i];
      R[i]  = h_R[i];
    }

    cudaFreeHost(h_RT); cudaFreeHost(h_R);
    cudaFree(d_RT);     cudaFree(d_R);
    cudaFree(d_b);      cudaFree(d_A);    cudaFree(d_t0);
    cudaFree(d_mean_v); cudaFree(d_sd_v);
    cudaFree(d_l); cudaFree(d_u); cudaFree(d_a); cudaFree(d_c); cudaFree(d_n);
    free(h_c); free(h_l); free(h_u); free(h_a); free(h_n);
    free(h_b);      free(h_A);    free(h_t0);
    free(h_mean_v); free(h_sd_v);
}


