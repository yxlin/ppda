//#include <unistd.h>
//#include <stdio.h>  // C printing
#include <curand_kernel.h> // Device random API
//#include "../inst/include/common.h"  
#include "../inst/include/util.h"
#include "../inst/include/reduce.h"  
#include <armadillo> // Armadillo vector operations

extern "C" void runif_entry(int *n, double *min, double *max, int *nth, 
  bool *dp,double *out);
extern "C" void rnorm_entry(int *n, double *mean, double *sd, int *nth, 
  bool *dp, double *out);
extern "C" void rtnorm_entry(int *n, double *mean, double *sd, double *l,
  double *u, int *nth, bool *dp, double *out);
extern "C" void rlbad_entry(int *n, double *b, double *A, double *mean_v,
  int *nmean_v, double *sd_v, int *nsd_v, double *t0, int *nth, double *RT, 
  int *R);
extern "C" void rlbaf_entry(int *n, double *b, double *A, double *mean_v,
  int *nmean_v, double *sd_v, int *nsd_v, double *t0, int *nth, double *RT, 
  int *R);
extern "C" void rlbad_n1(int *n, double *b, double *A, double *mean_v,
  int *nmean_v, double *sd_v, int *nsd_v, double *t0, int *nth, double *RT, 
  int *R);
extern "C" void rlbaf_n1(int *n, double *b, double *A, double *mean_v,
  int *nmean_v, double *sd_v, int *nsd_v, double *t0, int *nth, double *RT, 
  int *R);

extern "C" void rplba1_entry(int *nsim, double *b, double *A, double *mean_v, int *nmean_v, 
  double *mean_w, double *sd_v, double *t0, double *T0, int *nth, int *R,
  double *RT);

extern "C" void rplba2_entry(int *nsim, double *b, double *A, double *mean_v, int *nmean_v, 
  double *mean_w, double *sd_v, double *sd_w,  double *t0, double *T0, int *nth, int *_R,
  double *RT);

extern "C" void rplba3_entry(int *nsim, double *b, double *A, double* c, double *mean_v,
               int *nmean_v, double *mean_w, double *sd_v, double *sd_w,
               double *t0, double *swt1, double *swt2, double *swtD, bool *a, int *nth,
               int *R, double *RT);

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

__global__ void rlba_kernel(unsigned int* n, double* b, double* A, 
  double* mean_v, double* sd_v, double* t0, double* lower, double* upper, 
  double* a, bool* c, double* RT, unsigned int* R)
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

__global__ void rlba_kernel(unsigned int* n, float* b, float* A, 
  float* mean_v, float* sd_v, float* t0, float* lower, float* upper, 
  float* a, bool* c, float* RT, unsigned int* R)
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
  double *mean_v, double *sd_v, double *t0, double *lower, double *upper, 
  double *a, bool *c, double *RT0, unsigned int *R) {
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

__global__ void rlba_n1_kernel(unsigned int *n, float *b, float *A,
                               float *mean_v, float *sd_v, float *t0,
                               float *lower, float *upper, float *a,
                               bool *c, float *RT0, unsigned int *R) {
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

__global__ void rplba1_kernel(unsigned int* n, float* b, float* A, 
  float* mean_v, float* mean_w, float* sd_v, float* t0, float* lv, float* uv, 
  float* av, bool* cv, float* lw, float* uw, float* aw, bool* cw, float *T0, 
  float* RT, unsigned int* R)
{
  const int numThreads = blockDim.x * gridDim.x;
  const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
  float z0, z1, x0, x1, v0, v1, w0, w1, DT_tmp;
  float dt0_stage1, dt1_stage1, dt0_stage2, dt1_stage2;
  unsigned int R_tmp;
    
  curandState_t state;
  curand_init( (clock64() << 20) + threadID, 0, 0, &state);
  for (size_t i = threadID; i <*n; i += numThreads)
  {
    v0 = cv[0] ? rtnorm0_device(&state, mean_v[0], sd_v[0], lv[0], uv[0]) :
                 rtnorm1_device(&state, av[0], mean_v[0], sd_v[0], lv[0], uv[0]);
    v1 = cv[1] ? rtnorm0_device(&state, mean_v[1], sd_v[1], lv[1], uv[1]) :
                 rtnorm1_device(&state, av[1], mean_v[1], sd_v[1], lv[1], uv[1]);
    w0 = cw[0] ? rtnorm0_device(&state, mean_w[0], sd_v[0], lw[0], uw[0]) :
                  rtnorm1_device(&state, aw[0], mean_w[0], sd_v[0], lw[0], uw[0]);
    w1 = cw[1] ? rtnorm0_device(&state, mean_w[1], sd_v[1], lw[1], uw[1]) :
                  rtnorm1_device(&state, aw[1], mean_w[1], sd_v[1], lw[1], uw[1]);
    x0 = A[0] * curand_uniform(&state); // Stage 1 starting pos choice 0
    x1 = A[0] * curand_uniform(&state); // Stage 1 starting pos choice 1
    z0 = x0 + (*T0)*v0; // Stage 2 starting pos choice 0
    z1 = x1 + (*T0)*v1; // Stage 2 starting pos choice 1
    dt0_stage1 = (b[0] - x0) / v0;
    dt1_stage1 = (b[0] - x1) / v1;
    DT_tmp = dt0_stage1 < dt1_stage1 ? dt0_stage1 : dt1_stage1;
    R_tmp  = dt0_stage1 < dt1_stage1 ? 1 : 2;

    if (DT_tmp < *T0) {
         RT[i] = DT_tmp + t0[0]; 
         R[i]  = R_tmp;
    } else {
        dt0_stage2 = (b[0] - z0) / w0;
        dt1_stage2 = (b[0] - z1) / w1;
        RT[i] = dt0_stage2 < dt1_stage2 ? dt0_stage2 + T0[0] + t0[0] : dt1_stage2 + T0[0] + t0[0];
        R[i]  = dt0_stage2 < dt1_stage2 ? 1 : 2;
    }
  }
}


__global__ void rplba1_n1_kernel(unsigned int* n, float* b, float* A, 
  float* mean_v, float* mean_w, float* sd_v, float* t0, float* lv, float* uv, 
  float* av, bool* cv, float* lw, float* uw, float* aw, bool* cw, float *T0, 
  float* RT, unsigned int* R)
{
  const int numThreads = blockDim.x * gridDim.x;
  const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
  float z0, z1, x0, x1, v0, v1, w0, w1, DT_tmp;
  float dt0_stage1, dt1_stage1, dt0_stage2, dt1_stage2;
  unsigned int R_tmp;
    
  curandState_t state;
  curand_init( (clock64() << 20) + threadID, 0, 0, &state);
  for (size_t i = threadID; i <*n; i += numThreads)
  {
    v0 = cv[0] ? rtnorm0_device(&state, mean_v[0], sd_v[0], lv[0], uv[0]) :
                 rtnorm1_device(&state, av[0], mean_v[0], sd_v[0], lv[0], uv[0]);
    v1 = cv[1] ? rtnorm0_device(&state, mean_v[1], sd_v[1], lv[1], uv[1]) :
                 rtnorm1_device(&state, av[1], mean_v[1], sd_v[1], lv[1], uv[1]);
    w0 = cw[0] ? rtnorm0_device(&state, mean_w[0], sd_v[0], lw[0], uw[0]) :
                  rtnorm1_device(&state, aw[0], mean_w[0], sd_v[0], lw[0], uw[0]);
    w1 = cw[1] ? rtnorm0_device(&state, mean_w[1], sd_v[1], lw[1], uw[1]) :
                  rtnorm1_device(&state, aw[1], mean_w[1], sd_v[1], lw[1], uw[1]);
    x0 = A[0] * curand_uniform(&state); // Stage 1 starting pos choice 0
    x1 = A[0] * curand_uniform(&state); // Stage 1 starting pos choice 1
    z0 = x0 + (*T0)*v0; // Stage 2 starting pos choice 0
    z1 = x1 + (*T0)*v1; // Stage 2 starting pos choice 1
    dt0_stage1 = (b[0] - x0) / v0;
    dt1_stage1 = (b[0] - x1) / v1;
    DT_tmp = dt0_stage1 < dt1_stage1 ? dt0_stage1 : 0;
    R_tmp  = dt0_stage1 < dt1_stage1 ? 1 : 2;

    if (DT_tmp < *T0) {
        RT[i] = (DT_tmp != 0) ? DT_tmp + t0[0] : 0; 
        R[i]  = R_tmp;
    } else {
        dt0_stage2 = (b[0] - z0) / w0;
        dt1_stage2 = (b[0] - z1) / w1;
        RT[i] = dt0_stage2 < dt1_stage2 ? dt0_stage2 + T0[0] + t0[0] : 0;
        R[i]  = dt0_stage2 < dt1_stage2 ? 1 : 2;
    }
  }
}


__global__ void rplba2_n1_kernel(unsigned int* n, float* b, float* A, 
  float* mean_v, float* mean_w, float* sd_v, float* sd_w, float* t0, float* lv, float* uv, 
  float* av, bool* cv, float* lw, float* uw, float* aw, bool* cw, float *T0, 
  float* RT, unsigned int* R)
{
  const int numThreads = blockDim.x * gridDim.x;
  const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
  float z0, z1, x0, x1, v0, v1, w0, w1, DT_tmp;
  float dt0_stage1, dt1_stage1, dt0_stage2, dt1_stage2;
  unsigned int R_tmp;
    
  curandState_t state;
  curand_init( (clock64() << 20) + threadID, 0, 0, &state);
  for (size_t i = threadID; i <*n; i += numThreads)
  {
    v0 = cv[0] ? rtnorm0_device(&state, mean_v[0], sd_v[0], lv[0], uv[0]) :
                 rtnorm1_device(&state, av[0], mean_v[0], sd_v[0], lv[0], uv[0]);
    v1 = cv[1] ? rtnorm0_device(&state, mean_v[1], sd_v[1], lv[1], uv[1]) :
                 rtnorm1_device(&state, av[1], mean_v[1], sd_v[1], lv[1], uv[1]);
    w0 = cw[0] ? rtnorm0_device(&state, mean_w[0], sd_v[0], lw[0], uw[0]) :
                 rtnorm1_device(&state, aw[0], mean_w[0], sd_w[0], lw[0], uw[0]);
    w1 = cw[1] ? rtnorm0_device(&state, mean_w[1], sd_v[1], lw[1], uw[1]) :
                 rtnorm1_device(&state, aw[1], mean_w[1], sd_w[1], lw[1], uw[1]);
    x0 = A[0] * curand_uniform(&state); // Stage 1 starting pos choice 0
    x1 = A[1] * curand_uniform(&state); // Stage 1 starting pos choice 1
    z0 = x0 + (*T0)*v0; // Stage 2 starting pos choice 0
    z1 = x1 + (*T0)*v1; // Stage 2 starting pos choice 1
    dt0_stage1 = (b[0] - x0) / v0;
    dt1_stage1 = (b[1] - x1) / v1;
    //DT_tmp = dt0_stage1 < dt1_stage1 ? dt0_stage1 : dt1_stage1;
    DT_tmp = dt0_stage1 < dt1_stage1 ? dt0_stage1 : 0;
    R_tmp  = dt0_stage1 < dt1_stage1 ? 1 : 2;

    if (DT_tmp < *T0) {
        RT[i] = (DT_tmp != 0) ? DT_tmp + t0[0] : 0; 
        // RT[i] = DT_tmp + t0[0]; 
         R[i]  = R_tmp;
    } else {
        dt0_stage2 = (b[0] - z0) / w0;
        dt1_stage2 = (b[0] - z1) / w1;
        //RT[i] = dt0_stage2 < dt1_stage2 ? dt0_stage2 + T0[0] + t0[0] : dt1_stage2 + T0[0] + t0[0];
        RT[i] = dt0_stage2 < dt1_stage2 ? dt0_stage2 + T0[0] + t0[0] : 0;
        R[i]  = dt0_stage2 < dt1_stage2 ? 1 : 2;
    }
  }
}

__global__ void rplba2_kernel(unsigned int* n, float* b, float* A, 
  float* mean_v, float* mean_w, float* sd_v, float* sd_w, float* t0, float* lv, float* uv, 
  float* av, bool* cv, float* lw, float* uw, float* aw, bool* cw, float *T0, 
  float* RT, unsigned int* R)
{
  const int numThreads = blockDim.x * gridDim.x;
  const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
  float z0, z1, x0, x1, v0, v1, w0, w1, DT_tmp;
  float dt0_stage1, dt1_stage1, dt0_stage2, dt1_stage2;
  unsigned int R_tmp;
    
  curandState_t state;
  curand_init( (clock64() << 20) + threadID, 0, 0, &state);
  for (size_t i = threadID; i <*n; i += numThreads)
  {
    v0 = cv[0] ? rtnorm0_device(&state, mean_v[0], sd_v[0], lv[0], uv[0]) :
                 rtnorm1_device(&state, av[0], mean_v[0], sd_v[0], lv[0], uv[0]);
    v1 = cv[1] ? rtnorm0_device(&state, mean_v[1], sd_v[1], lv[1], uv[1]) :
                 rtnorm1_device(&state, av[1], mean_v[1], sd_v[1], lv[1], uv[1]);
    w0 = cw[0] ? rtnorm0_device(&state, mean_w[0], sd_v[0], lw[0], uw[0]) :
                 rtnorm1_device(&state, aw[0], mean_w[0], sd_w[0], lw[0], uw[0]);
    w1 = cw[1] ? rtnorm0_device(&state, mean_w[1], sd_v[1], lw[1], uw[1]) :
                 rtnorm1_device(&state, aw[1], mean_w[1], sd_w[1], lw[1], uw[1]);
    x0 = A[0] * curand_uniform(&state); // Stage 1 starting pos choice 0
    x1 = A[1] * curand_uniform(&state); // Stage 1 starting pos choice 1
    z0 = x0 + (*T0)*v0; // Stage 2 starting pos choice 0
    z1 = x1 + (*T0)*v1; // Stage 2 starting pos choice 1
    dt0_stage1 = (b[0] - x0) / v0;
    dt1_stage1 = (b[1] - x1) / v1;
    DT_tmp = dt0_stage1 < dt1_stage1 ? dt0_stage1 : dt1_stage1;
    R_tmp  = dt0_stage1 < dt1_stage1 ? 1 : 2;

    if (DT_tmp < *T0) {
         RT[i] = DT_tmp + t0[0]; 
         R[i]  = R_tmp;
    } else {
        dt0_stage2 = (b[0] - z0) / w0;
        dt1_stage2 = (b[0] - z1) / w1;
        RT[i] = dt0_stage2 < dt1_stage2 ? dt0_stage2 + T0[0] + t0[0] : dt1_stage2 + T0[0] + t0[0];
        R[i]  = dt0_stage2 < dt1_stage2 ? 1 : 2;
    }
  }
}

__global__ void rplba3_kernel(unsigned int* n, float* b, float* A, float* c,
  bool* a, float* mean_v, float* mean_w, float* sd_v, float* sd_w, float* t0, float* lv, float* uv,  
  float* av, bool* cv, float* lw, float* uw, float* aw,
  bool* cw, float *swt1, float* swt2, float *swtD,
  float* RT, unsigned int* R)
{
  const int numThreads = blockDim.x * gridDim.x;
  const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
  float u0, u1, v0, v1, w0, w1, Y0, Y1, Z0, Z1;
  float dt0_stage1, dt0_stage2, dt0_stage3, dt1_stage1, dt1_stage2, dt1_stage3;
  float x0, x1, v0_tmp, v1_tmp;
  float DT_tmp1, DT_tmp2, DT_tmp3;
  unsigned int R_tmp1, R_tmp2,  R_tmp3;
    
  curandState_t state;
  curand_init( (clock64() << 20) + threadID, 0, 0, &state);
  for (size_t i = threadID; i <*n; i += numThreads)
  {
   // Stage 1
    u0 = cv[0] ? rtnorm0_device(&state, mean_v[0], sd_v[0], lv[0], uv[0]) :
                 rtnorm1_device(&state, av[0], mean_v[0], sd_v[0], lv[0], uv[0]);
    u1 = cv[1] ? rtnorm0_device(&state, mean_v[1], sd_v[1], lv[1], uv[1]) :
                 rtnorm1_device(&state, av[1], mean_v[1], sd_v[1], lv[1], uv[1]);
    x0 = A[0] * curand_uniform(&state);
    x1 = A[1] * curand_uniform(&state);
    dt0_stage1 = (b[0] - x0) / u0;
    dt1_stage1 = (b[1] - x1) / u1;
    DT_tmp1 = dt0_stage1 < dt1_stage1 ? dt0_stage1: dt1_stage1; 
    R_tmp1  = dt0_stage1 < dt1_stage1 ? 1 : 2;

    // Stage 2
    Y0 = a[2] ? b[0] - (x0 + *swt1*u0) : c[0] - (x0 + *swt1*u0);
    Y1 = a[2] ? b[1] - (x1 + *swt1*u1) : c[1] - (x1 + *swt1*u1);
    v0_tmp = cw[0] ? rtnorm0_device(&state, mean_w[0], sd_w[0], lw[0], uw[0]) :
                     rtnorm1_device(&state, aw[0], mean_w[0], sd_w[0], lw[0], uw[0]);
    v1_tmp = cw[1] ? rtnorm0_device(&state, mean_w[1], sd_w[1], lw[1], uw[1]) :
                     rtnorm1_device(&state, aw[1], mean_w[1], sd_w[1], lw[1], uw[1]);

    v0 = a[1] ? u0 : v0_tmp;
    v1 = a[1] ? u1 : v1_tmp;
    dt0_stage2 = Y0 / v0;
    dt1_stage2 = Y1 / v1;
    DT_tmp2 = dt0_stage2 < dt1_stage2 ? dt0_stage2 + *swt1 : dt1_stage2 + *swt1;
    R_tmp2  = dt0_stage2 < dt1_stage2 ? 1 : 2;

    // Stage 3
    Z0 = a[1] ? Y0 - *swtD*v0 : (Y0 - *swtD*v0) + c[0] - b[0];
    Z1 = a[1] ? Y1 - *swtD*v1 : (Y1 - *swtD*v1) + c[1] - b[1];
    v0_tmp = cw[0] ? rtnorm0_device(&state, mean_w[0], sd_w[0], lw[0], uw[0]) :
                     rtnorm1_device(&state, aw[0], mean_w[0], sd_w[0], lw[0], uw[0]);
    v1_tmp = cw[1] ? rtnorm0_device(&state, mean_w[1], sd_w[1], lw[1], uw[1]) :
                     rtnorm1_device(&state, aw[1], mean_w[1], sd_w[1], lw[1], uw[1]);
    w0 = a[1] ? v0_tmp : v0;
    w1 = a[1] ? v1_tmp : v1;
    dt0_stage3 = Z0 / w0;
    dt1_stage3 = Z1 / w1;
    DT_tmp3 = dt0_stage3 < dt1_stage3 ? dt0_stage3 + *swt2 : dt1_stage3 + *swt2;
    R_tmp3  = dt0_stage3 < dt1_stage3 ? 1 : 2;

    if (DT_tmp1 <= *swt1) {
      RT[i] = DT_tmp1 + *t0;
      R[i]  = R_tmp1;
    } else if (DT_tmp2 <= *swt2 || a[0]) {
      RT[i] = DT_tmp2 + *t0;
      R[i]  = R_tmp2;
    } else {
      RT[i] = DT_tmp3 + *t0;
      R[i]  = R_tmp3;
    }
  }
}


__global__ void rplba3_n1_kernel(unsigned int* n, float* b, float* A, float* c,
  bool* a, float* mean_v, float* mean_w, float* sd_v, float* sd_w, float* t0, float* lv, float* uv,  
  float* av, bool* cv, float* lw, float* uw, float* aw,
  bool* cw, float *swt1, float* swt2, float *swtD,
  float* RT, unsigned int* R)
{
  const int numThreads = blockDim.x * gridDim.x;
  const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
  float u0, u1, v0, v1, w0, w1, Y0, Y1, Z0, Z1;
  float dt0_stage1, dt0_stage2, dt0_stage3, dt1_stage1, dt1_stage2, dt1_stage3;
  float x0, x1, v0_tmp, v1_tmp;
  float DT_tmp1, DT_tmp2, DT_tmp3;
  unsigned int R_tmp1, R_tmp2,  R_tmp3;
    
  curandState_t state;
  curand_init( (clock64() << 20) + threadID, 0, 0, &state);
  for (size_t i = threadID; i <*n; i += numThreads)
  {
   // Stage 1
    u0 = cv[0] ? rtnorm0_device(&state, mean_v[0], sd_v[0], lv[0], uv[0]) :
                 rtnorm1_device(&state, av[0], mean_v[0], sd_v[0], lv[0], uv[0]);
    u1 = cv[1] ? rtnorm0_device(&state, mean_v[1], sd_v[1], lv[1], uv[1]) :
                 rtnorm1_device(&state, av[1], mean_v[1], sd_v[1], lv[1], uv[1]);
    x0 = A[0] * curand_uniform(&state);
    x1 = A[1] * curand_uniform(&state);
    dt0_stage1 = (b[0] - x0) / u0;
    dt1_stage1 = (b[1] - x1) / u1;
    //DT_tmp1 = dt0_stage1 < dt1_stage1 ? dt0_stage1: dt1_stage1; 
    DT_tmp1 = dt0_stage1 < dt1_stage1 ? dt0_stage1: 0; 
    R_tmp1  = dt0_stage1 < dt1_stage1 ? 1 : 2;

    // Stage 2
    Y0 = a[2] ? b[0] - (x0 + *swt1*u0) : c[0] - (x0 + *swt1*u0);
    Y1 = a[2] ? b[1] - (x1 + *swt1*u1) : c[1] - (x1 + *swt1*u1);
    v0_tmp = cw[0] ? rtnorm0_device(&state, mean_w[0], sd_w[0], lw[0], uw[0]) :
                     rtnorm1_device(&state, aw[0], mean_w[0], sd_w[0], lw[0], uw[0]);
    v1_tmp = cw[1] ? rtnorm0_device(&state, mean_w[1], sd_w[1], lw[1], uw[1]) :
                     rtnorm1_device(&state, aw[1], mean_w[1], sd_w[1], lw[1], uw[1]);

    v0 = a[1] ? u0 : v0_tmp;
    v1 = a[1] ? u1 : v1_tmp;
    dt0_stage2 = Y0 / v0;
    dt1_stage2 = Y1 / v1;
    //DT_tmp2 = dt0_stage2 < dt1_stage2 ? dt0_stage2 + *swt1 : dt1_stage2 + *swt1;
    DT_tmp2 = dt0_stage2 < dt1_stage2 ? dt0_stage2 + *swt1 : 0;
    R_tmp2  = dt0_stage2 < dt1_stage2 ? 1 : 2;

    // Stage 3
    Z0 = a[1] ? Y0 - *swtD*v0 : (Y0 - *swtD*v0) + c[0] - b[0];
    Z1 = a[1] ? Y1 - *swtD*v1 : (Y1 - *swtD*v1) + c[1] - b[1];
    v0_tmp = cw[0] ? rtnorm0_device(&state, mean_w[0], sd_w[0], lw[0], uw[0]) :
                     rtnorm1_device(&state, aw[0], mean_w[0], sd_w[0], lw[0], uw[0]);
    v1_tmp = cw[1] ? rtnorm0_device(&state, mean_w[1], sd_w[1], lw[1], uw[1]) :
                     rtnorm1_device(&state, aw[1], mean_w[1], sd_w[1], lw[1], uw[1]);
    w0 = a[1] ? v0_tmp : v0;
    w1 = a[1] ? v1_tmp : v1;
    dt0_stage3 = Z0 / w0;
    dt1_stage3 = Z1 / w1;
    //DT_tmp3 = dt0_stage3 < dt1_stage3 ? dt0_stage3 + *swt2 : dt1_stage3 + *swt2;
    DT_tmp3 = dt0_stage3 < dt1_stage3 ? dt0_stage3 + *swt2 : 0;
    R_tmp3  = dt0_stage3 < dt1_stage3 ? 1 : 2;

    if (DT_tmp1 <= *swt1) {
      //RT[i] = DT_tmp1 + *t0;
      RT[i] = (DT_tmp1 != 0) ? DT_tmp1 + *t0 : 0; 
      R[i]  = R_tmp1;
    } else if (DT_tmp2 <= *swt2 || a[0]) {
      //RT[i] = DT_tmp2 + *t0;
      RT[i] = (DT_tmp2 != 0) ? DT_tmp2 + *t0 : 0; 
      R[i]  = R_tmp2;
    } else {
      //RT[i] = DT_tmp3 + *t0;
      RT[i] = (DT_tmp3 != 0) ? DT_tmp3 + *t0 : 0; 
      R[i]  = R_tmp3;
    }
  }
}

void runif_entry(int *n, double *min, double *max, int *nth, bool *dp, 
  double *out)
{
  if(*dp) {
    size_t ndSize = *n * sizeof(double);
    size_t dSize  = 1 * sizeof(double);
    double *d_min, *d_max, *d_out, *h_out;
    cudaMalloc((void**) &d_out,   ndSize);
    cudaMalloc((void**) &d_min,   dSize);
    cudaMalloc((void**) &d_max,   dSize);
    cudaHostAlloc((void**)&h_out, ndSize, cudaHostAllocDefault);
    cudaMemcpy(d_min, min, dSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_max, max, dSize, cudaMemcpyHostToDevice);
    runif_kernel<<<((*n)/(*nth) + 1), *nth>>>((*n), d_min, d_max, d_out);
    cudaMemcpy(h_out, d_out, ndSize, cudaMemcpyDeviceToHost);
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
    
    cudaMalloc((void**) &d_out,   nfSize);
    cudaMalloc((void**) &d_min,   fSize);
    cudaMalloc((void**) &d_max,   fSize);
    cudaHostAlloc((void**)&h_out, nfSize, cudaHostAllocDefault);
    cudaMemcpy(d_min, h_min, fSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_max, h_max, fSize, cudaMemcpyHostToDevice);
    runif_kernel<<<((*n)/(*nth) + 1), *nth>>>((*n), d_min, d_max, d_out);
    cudaMemcpy(h_out, d_out, nfSize, cudaMemcpyDeviceToHost);
    for(size_t i=0; i<*n; i++) { out[i] = (double)h_out[i]; }
    free(h_min); free(h_max);
    cudaFreeHost(h_out); cudaFree(d_out); cudaFree(d_min); cudaFree(d_max);
  }
}

void rnorm_entry(int *n, double *mean, double *sd, int *nth, bool *dp, 
  double *out)
{
  if (*dp) {
    double *d_out, *d_mean, *d_sd, *h_out;
    cudaHostAlloc((void**)&h_out, *n * sizeof(double), cudaHostAllocDefault);
    cudaMalloc((void**)&d_out,    *n * sizeof(double));
    cudaMalloc((void**)&d_mean,    1 * sizeof(double));
    cudaMalloc((void**)&d_sd,      1 * sizeof(double));
    cudaMemcpy(d_mean, mean,       1 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sd, sd,           1 * sizeof(double), cudaMemcpyHostToDevice);
    rnorm_kernel<<<((*n)/(*nth) + 1), *nth>>>(*n, d_mean, d_sd, d_out);
    cudaMemcpy(h_out, d_out, *n * sizeof(double), cudaMemcpyDeviceToHost);
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
    
    cudaHostAlloc((void**)&h_out, nfSize, cudaHostAllocDefault);
    cudaMalloc((void**)&d_out,  nfSize);
    cudaMalloc((void**)&d_mean, fSize);
    cudaMalloc((void**)&d_sd,   fSize);
    cudaMemcpy(d_mean, h_mean,  fSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sd,   h_sd,    fSize, cudaMemcpyHostToDevice);
    rnorm_kernel<<<((*n)/(*nth) + 1), *nth>>>(*n, d_mean, d_sd, d_out);
    cudaMemcpy(h_out, d_out, nfSize, cudaMemcpyDeviceToHost);
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
    cudaHostAlloc((void**)&h_out, ndSize, cudaHostAllocDefault);
    cudaMalloc((void**) &d_out,   ndSize);
    cudaMalloc((void**) &d_mean,  dSize);
    cudaMalloc((void**) &d_sd,    dSize);
    cudaMalloc((void**) &d_l,     dSize);
    cudaMalloc((void**) &d_u,     dSize);
    cudaMemcpy(d_mean, mean, dSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sd,     sd, dSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_l,  h_stdl, dSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u,  h_stdu, dSize, cudaMemcpyHostToDevice);
    
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
    
    cudaHostAlloc((void**)&h_out, nfSize, cudaHostAllocDefault);
    cudaMalloc((void**) &d_out,   nfSize);
    cudaMalloc((void**) &d_mean,  fSize);
    cudaMalloc((void**) &d_sd,    fSize);
    cudaMalloc((void**) &d_l,     fSize);
    cudaMalloc((void**) &d_u,     fSize);
    cudaMemcpy(d_mean, h_mean, fSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sd,   h_sd,   fSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_l,  h_l,      fSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u,  h_u,      fSize, cudaMemcpyHostToDevice);
    
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
    cudaFree(d_mean); cudaFree(d_sd); cudaFree(d_l); cudaFree(d_u);
  }
}

void rlbad_entry(int *n, double *b,double *A, double *mean_v, 
  int *nmean_v, double *sd_v, int *nsd_v, double *t0, int *nth, double *RT,
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
  
  cudaHostAlloc((void**)&h_RT,   ndSize, cudaHostAllocDefault);
  cudaHostAlloc((void**)&h_R,    nuSize, cudaHostAllocDefault);
  cudaMalloc((void**) &d_R,      nuSize);
  cudaMalloc((void**) &d_RT,     ndSize);
  cudaMalloc((void**) &d_n,      uSize);
  cudaMalloc((void**) &d_b,      dSize);
  cudaMalloc((void**) &d_A,      dSize);
  cudaMalloc((void**) &d_t0,     dSize);
  cudaMalloc((void**) &d_sd_v,   vdSize);
  cudaMalloc((void**) &d_mean_v, vdSize);
  cudaMalloc((void**) &d_l,      vdSize);
  cudaMalloc((void**) &d_u,      vdSize);
  cudaMalloc((void**) &d_a,      vdSize);
  cudaMalloc((void**) &d_c,      vbSize);
  
  cudaMemcpy(d_n, h_n,  uSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b,   b,  dSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_A,   A,  dSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_t0, t0,  dSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mean_v, mean_v, vdSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_sd_v,     sd_v, vdSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_l,         h_l, vdSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_u,         h_u, vdSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_a,         h_a, vdSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c,         h_c, vbSize, cudaMemcpyHostToDevice);
  
  rlba_kernel<<<((*n)/(*nth) + 1), *nth>>>(d_n, d_b, d_A, d_mean_v, d_sd_v,
    d_t0, d_l, d_u, d_a, d_c, d_RT,
    d_R);
  cudaMemcpy(h_RT, d_RT, ndSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_R,  d_R,  nuSize, cudaMemcpyDeviceToHost);
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

void rlbaf_entry(int *n, double *b,double *A, double *mean_v, int *nmean_v,
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
  
  cudaHostAlloc((void**)&h_RT,   nfSize, cudaHostAllocDefault);
  cudaHostAlloc((void**)&h_R,    nuSize, cudaHostAllocDefault);
  cudaMalloc((void**) &d_R,      nuSize);
  cudaMalloc((void**) &d_RT,     nfSize);
  cudaMalloc((void**) &d_n,      uSize);
  cudaMalloc((void**) &d_b,      fSize);
  cudaMalloc((void**) &d_A,      fSize);
  cudaMalloc((void**) &d_t0,     fSize);
  cudaMalloc((void**) &d_sd_v,   vfSize);
  cudaMalloc((void**) &d_mean_v, vfSize);
  cudaMalloc((void**) &d_l,      vfSize);
  cudaMalloc((void**) &d_u,      vfSize);
  cudaMalloc((void**) &d_a,      vfSize);
  cudaMalloc((void**) &d_c,      vbSize);
  cudaMemcpy(d_n, h_n,  uSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b,  fSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_A, h_A,  fSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_t0,h_t0, fSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mean_v, h_mean_v, vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_sd_v,   h_sd_v,   vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_l,         h_l,   vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_u,         h_u,   vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_a,         h_a,   vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c,         h_c,   vbSize, cudaMemcpyHostToDevice);
  
  rlba_kernel<<<((*n)/(*nth) + 1), *nth>>>(d_n, d_b, d_A, d_mean_v, d_sd_v,
    d_t0, d_l, d_u, d_a, d_c, d_RT,
    d_R);
  cudaMemcpy(h_RT, d_RT, nfSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_R,  d_R,  nuSize, cudaMemcpyDeviceToHost);
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
  free(h_b);      free(h_A);    free(h_t0); free(h_mean_v); free(h_sd_v);
}

void rlbad_n1(int *n, double *b,double *A, double *mean_v, int *nmean_v,
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
  
  cudaHostAlloc((void**)&h_RT,   ndSize, cudaHostAllocDefault);
  cudaHostAlloc((void**)&h_R,    nuSize, cudaHostAllocDefault);
  cudaMalloc((void**) &d_RT,     ndSize);
  cudaMalloc((void**) &d_R,      nuSize);
  cudaMalloc((void**) &d_n,      uSize);
  cudaMalloc((void**) &d_b,      dSize);
  cudaMalloc((void**) &d_A,      dSize);
  cudaMalloc((void**) &d_t0,     dSize);
  cudaMalloc((void**) &d_sd_v,   vdSize);
  cudaMalloc((void**) &d_mean_v, vdSize);
  cudaMalloc((void**) &d_l,      vdSize);
  cudaMalloc((void**) &d_u,      vdSize);
  cudaMalloc((void**) &d_a,      vdSize);
  cudaMalloc((void**) &d_c,      vbSize);
  cudaMemcpy(d_n, h_n,  uSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b,   b,  dSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_A,   A,  dSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_t0, t0,  dSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mean_v, mean_v, vdSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_sd_v,     sd_v, vdSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_l,         h_l, vdSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_u,         h_u, vdSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_a,         h_a, vdSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c,         h_c, vbSize, cudaMemcpyHostToDevice);
  
  rlba_n1_kernel<<<((*n)/(*nth) + 1), *nth>>>(d_n, d_b, d_A, d_mean_v, d_sd_v,
    d_t0, d_l, d_u, d_a, d_c, d_RT, d_R);
  cudaMemcpy(h_RT, d_RT, ndSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_R,  d_R,  nuSize, cudaMemcpyDeviceToHost);
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

void rlbaf_n1(int *n, double *b, double *A, double *mean_v, int *nmean_v,
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
  
  cudaHostAlloc((void**)&h_RT,   nfSize, cudaHostAllocDefault);
  cudaHostAlloc((void**)&h_R,    nuSize, cudaHostAllocDefault);
  cudaMalloc((void**) &d_RT,     nfSize);
  cudaMalloc((void**) &d_R,      nuSize);
  cudaMalloc((void**) &d_n,      uSize);
  cudaMalloc((void**) &d_b,      fSize);
  cudaMalloc((void**) &d_A,      fSize);
  cudaMalloc((void**) &d_t0,     fSize);
  cudaMalloc((void**) &d_sd_v,   vfSize);
  cudaMalloc((void**) &d_mean_v, vfSize);
  cudaMalloc((void**) &d_l,      vfSize);
  cudaMalloc((void**) &d_u,      vfSize);
  cudaMalloc((void**) &d_a,      vfSize);
  cudaMalloc((void**) &d_c,      vbSize);
  cudaMemcpy(d_n, h_n,  uSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b,  fSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_A, h_A,  fSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_t0,h_t0, fSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mean_v,h_mean_v,vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_sd_v,    h_sd_v,vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_l,         h_l, vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_u,         h_u, vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_a,         h_a, vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c,         h_c, vbSize, cudaMemcpyHostToDevice);
  
  rlba_n1_kernel<<<((*n)/(*nth) + 1), *nth>>>(d_n, d_b, d_A, d_mean_v, d_sd_v,
    d_t0, d_l, d_u, d_a, d_c, d_RT,
    d_R);
  cudaMemcpy(h_RT, d_RT, nfSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_R,  d_R,  nuSize, cudaMemcpyDeviceToHost);
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
  free(h_b);      free(h_A);    free(h_t0); free(h_mean_v); free(h_sd_v);
}

void rn1(int *nsim, double *b, double *A, double *mean_v, int *nmean_v, 
  double *sd_v, double *t0, int *nth, unsigned int *d_R, float *d_RT0)
{
  
  bool *h_c, *d_c; // c for choice switch for rtnorm insider rlba_gpu
  float *d_b, *d_A, *d_mean_v, *d_sd_v, *d_t0, *d_l, *d_u, *d_a;
  float *h_b, *h_A, *h_mean_v, *h_sd_v, *h_t0, *h_l, *h_u, *h_a; // *h_a and *d_a stands for alphaStart
  unsigned int *d_nsim, *h_nsim;
  size_t uSize  = 1 * sizeof(unsigned int);
  size_t vfSize = nmean_v[0] * sizeof(float);
  size_t fSize  = 1 * sizeof(float);
  size_t vbSize = nmean_v[0] * sizeof(bool);
  h_c = (bool *)malloc(vbSize);
  h_l = (float *)malloc(vfSize);
  h_u = (float *)malloc(vfSize);
  h_a = (float *)malloc(vfSize);
  
  h_nsim   = (unsigned int *)malloc(uSize);
  h_b      = (float *)malloc(fSize);
  h_A      = (float *)malloc(fSize);
  h_t0     = (float *)malloc(fSize);
  h_mean_v = (float *)malloc(vfSize);
  h_sd_v   = (float *)malloc(vfSize);
  
  *h_nsim  = (unsigned int)*nsim;
  *h_b  = (float)*b;
  *h_A  = (float)*A;
  *h_t0 = (float)*t0;
  
  for(size_t i=0; i<nmean_v[0]; i++) {
    h_mean_v[i] = (float)mean_v[i];
    h_sd_v[i]   = (float)sd_v[i];
    h_l[i] = (0 - h_mean_v[i]) / h_sd_v[i]; // convert to mean=0, sd=1
    h_u[i] = (INFINITY - mean_v[i]) / h_sd_v[i]; // Should also be infinity
    h_a[i] = 0.5 * (std::sqrt(h_l[i]*h_l[i] + 4.0) + h_l[i]); // use in rtnorm1_device, alphaStar must be greater than 0
    h_c[i] = (h_l[i] < 0 && h_u[i]==INFINITY) || (h_l[i]==-INFINITY && h_u[i]) ||
      (isfinite(h_l[i]) && isfinite(h_u[i]) && h_l[i] < 0 && h_u[i] > 0 && ((h_u[i] - h_l[i]) > SQRT_2PI));
  }
  
  cudaMalloc((void**) &d_nsim, uSize);
  cudaMalloc((void**) &d_b,  fSize);
  cudaMalloc((void**) &d_A,  fSize);
  cudaMalloc((void**) &d_t0, fSize);
  cudaMalloc((void**) &d_mean_v, vfSize);
  cudaMalloc((void**) &d_sd_v,   vfSize);
  cudaMalloc((void**) &d_l,      vfSize);
  cudaMalloc((void**) &d_u,      vfSize);
  cudaMalloc((void**) &d_a,      vfSize);
  cudaMalloc((void**) &d_c,      vbSize);
  cudaMemcpy(d_nsim,   h_nsim,   uSize,  cudaMemcpyHostToDevice);
  cudaMemcpy(d_b,      h_b,      fSize,  cudaMemcpyHostToDevice);
  cudaMemcpy(d_A,      h_A,      fSize,  cudaMemcpyHostToDevice);
  cudaMemcpy(d_t0,     h_t0,     fSize,  cudaMemcpyHostToDevice);
  cudaMemcpy(d_mean_v, h_mean_v, vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_sd_v,   h_sd_v,   vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_l,      h_l,      vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_u,      h_u,      vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_a,      h_a,      vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c,      h_c,      vbSize, cudaMemcpyHostToDevice);
  
  rlba_n1_kernel<<<(*nsim)/(*nth), *nth>>>(d_nsim, d_b,  d_A, d_mean_v,
    d_sd_v, d_t0, d_l, d_u, d_a, d_c, d_RT0, d_R);
  free(h_b);
  free(h_A);
  free(h_mean_v);
  free(h_sd_v);
  free(h_t0);
  free(h_c);
  free(h_l);
  free(h_u);
  free(h_a);
  free(h_nsim);
  
  cudaFree(d_b);
  cudaFree(d_A);
  cudaFree(d_t0);
  cudaFree(d_mean_v);
  cudaFree(d_sd_v);
  cudaFree(d_l);
  cudaFree(d_u);
  cudaFree(d_a);
  cudaFree(d_c);
  cudaFree(d_nsim);
}

void rplba1_entry(int *nsim, double *b, double *A, double *mean_v, int *nmean_v, 
  double *mean_w, double *sd_v, double *t0, double *T0, int *nth, int *R,
  double *RT)
{
  bool *h_cv, *d_cv, *h_cw, *d_cw; 
  float *d_b, *d_A, *d_mean_v, *d_sd_v, *d_t0, *d_lv, *d_uv, *d_av;
  float *h_b, *h_A, *h_mean_v, *h_sd_v, *h_t0, *h_lv, *h_uv, *h_av; 
  float *d_T0, *d_mean_w, *d_lw, *d_uw, *d_aw;
  float *h_T0, *h_mean_w, *h_sd_w, *h_lw, *h_uw, *h_aw;
  float *h_RT, *d_RT;
  unsigned int *h_R, *d_R;

  unsigned int *d_nsim, *h_nsim;
  size_t uSize  = 1 * sizeof(unsigned int);
  size_t fSize  = 1 * sizeof(float);
  size_t nuSize  = *nsim * sizeof(unsigned int);
  size_t nfSize  = *nsim * sizeof(float);
  size_t vfSize = nmean_v[0] * sizeof(float);
  size_t vbSize = nmean_v[0] * sizeof(bool);
  
  h_cv = (bool  *)malloc(vbSize);
  h_lv = (float *)malloc(vfSize);
  h_uv = (float *)malloc(vfSize);
  h_av = (float *)malloc(vfSize);

  h_cw = (bool  *)malloc(vbSize);
  h_lw = (float *)malloc(vfSize);
  h_uw = (float *)malloc(vfSize);
  h_aw = (float *)malloc(vfSize);

  h_nsim   = (unsigned int *)malloc(uSize);
  h_b      = (float *)malloc(fSize);
  h_A      = (float *)malloc(fSize);
  h_t0     = (float *)malloc(fSize);
  h_T0     = (float *)malloc(fSize);
  h_mean_v = (float *)malloc(vfSize);
  h_mean_w = (float *)malloc(vfSize);
  h_sd_v   = (float *)malloc(vfSize);
  h_sd_w   = (float *)malloc(vfSize);
  
  *h_nsim  = (unsigned int)*nsim;
  *h_b  = (float)*b;
  *h_A  = (float)*A;
  *h_t0 = (float)*t0;
  *h_T0 = (float)*T0;

  for(size_t i=0; i<nmean_v[0]; i++) {
    h_mean_v[i] = (float)mean_v[i];
    h_sd_v[i]   = (float)sd_v[i];
    h_lv[i] = (0 - h_mean_v[i]) / h_sd_v[i]; 
    h_uv[i] = (INFINITY - h_mean_v[i]) / h_sd_v[i]; 
    h_av[i] = 0.5 * (std::sqrt(h_lv[i]*h_lv[i] + 4.0) + h_lv[i]); 
    h_cv[i] = (h_lv[i] < 0 && h_uv[i] == INFINITY) || (h_lv[i] == -INFINITY && h_uv[i]) ||
      (isfinite(h_lv[i]) && isfinite(h_uv[i]) && h_lv[i] < 0 && h_uv[i] > 0 && 
      ((h_uv[i] - h_lv[i]) > SQRT_2PI));

    h_mean_w[i] = (float)mean_w[i];
    h_sd_w[i]   = (float)sd_v[i];
    h_lw[i] = (0 - h_mean_w[i]) / h_sd_w[i]; 
    h_uw[i] = (INFINITY - h_mean_w[i]) / h_sd_w[i]; 
    h_aw[i] = 0.5 * (std::sqrt(h_lw[i]*h_lw[i] + 4.0) + h_lw[i]); 
    h_cw[i] = (h_lw[i] < 0 && h_uw[i]==INFINITY) || (h_lw[i]==-INFINITY && h_uw[i]) ||
      (isfinite(h_lw[i]) && isfinite(h_uw[i]) && h_lw[i] < 0 && h_uw[i] > 0 && 
      ((h_uw[i] - h_lw[i]) > SQRT_2PI));
  }

  cudaHostAlloc((void**)&h_RT,   nfSize, cudaHostAllocDefault);
  cudaHostAlloc((void**)&h_R,    nuSize, cudaHostAllocDefault);
  cudaMalloc((void**) &d_RT, nuSize);
  cudaMalloc((void**) &d_R,  nfSize);
  
  cudaMalloc((void**) &d_nsim, uSize);
  cudaMalloc((void**) &d_b,  fSize);
  cudaMalloc((void**) &d_A,  fSize);
  cudaMalloc((void**) &d_t0, fSize);
  cudaMalloc((void**) &d_T0, fSize);
  cudaMalloc((void**) &d_mean_v, vfSize);
  cudaMalloc((void**) &d_mean_w, vfSize);
  cudaMalloc((void**) &d_sd_v,   vfSize);
  cudaMalloc((void**) &d_lv,     vfSize);
  cudaMalloc((void**) &d_uv,     vfSize);
  cudaMalloc((void**) &d_av,     vfSize);
  cudaMalloc((void**) &d_cv,     vbSize);
  cudaMalloc((void**) &d_lw,     vfSize);
  cudaMalloc((void**) &d_uw,     vfSize);
  cudaMalloc((void**) &d_aw,     vfSize);
  cudaMalloc((void**) &d_cw,     vbSize);
  cudaMemcpy(d_nsim,   h_nsim,   uSize,  cudaMemcpyHostToDevice);
  cudaMemcpy(d_b,      h_b,      fSize,  cudaMemcpyHostToDevice);
  cudaMemcpy(d_A,      h_A,      fSize,  cudaMemcpyHostToDevice);
  cudaMemcpy(d_t0,     h_t0,     fSize,  cudaMemcpyHostToDevice);
  cudaMemcpy(d_T0,     h_T0,     fSize,  cudaMemcpyHostToDevice);
  cudaMemcpy(d_mean_v, h_mean_v, vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mean_w, h_mean_w, vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_sd_v,   h_sd_v,   vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_lv,     h_lv,    vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_uv,     h_uv,    vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_av,     h_av,    vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_cv,     h_cv,    vbSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_lw,     h_lw,    vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_uw,     h_uw,    vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_aw,     h_aw,    vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_cw,     h_cw,    vbSize, cudaMemcpyHostToDevice);
  
  rplba1_kernel<<<(*nsim)/(*nth), *nth>>>(d_nsim, d_b,  d_A, d_mean_v, d_mean_w,
    d_sd_v, d_t0, d_lv, d_uv, d_av, d_cv, d_lw, d_uw, d_aw, d_cw, d_T0, d_RT, d_R);

  cudaMemcpy(h_RT, d_RT, nfSize, cudaMemcpyDeviceToHost); cudaFree(d_RT);
  cudaMemcpy(h_R,  d_R,  nuSize, cudaMemcpyDeviceToHost); cudaFree(d_R);
  for(size_t i=0; i<*nsim; i++) {
      RT[i] = (double)h_RT[i];
      R[i]  = (int)h_R[i];
  }
  cudaFreeHost(h_RT);
  cudaFreeHost(h_R);

  free(h_b); 
  free(h_A);
  free(h_mean_v);
  free(h_mean_w);
  free(h_sd_v);
  free(h_t0);
  free(h_cv);
  free(h_lv);
  free(h_uv);
  free(h_av);
  free(h_cw);
  free(h_lw);
  free(h_uw);
  free(h_aw);
  free(h_nsim);
  free(h_T0);
  
  cudaFree(d_b);
  cudaFree(d_A);
  cudaFree(d_t0);
  cudaFree(d_mean_v);
  cudaFree(d_mean_w);
  cudaFree(d_sd_v);
  cudaFree(d_lv);
  cudaFree(d_uv);
  cudaFree(d_av);
  cudaFree(d_cv);
  
  cudaFree(d_lw);
  cudaFree(d_uw);
  cudaFree(d_aw);
  cudaFree(d_cw);
  
  cudaFree(d_nsim);
  cudaFree(d_T0);
}


void rplba1_n1(int *nsim, double *b, double *A, double *mean_v, int *nmean_v, 
  double *mean_w, double *sd_v, double *t0, double *T0, int *nth, unsigned int *d_R,
  float *d_RT)
{
  bool *h_cv, *d_cv, *h_cw, *d_cw; 
  float *d_b, *d_A, *d_mean_v, *d_sd_v, *d_t0, *d_lv, *d_uv, *d_av;
  float *h_b, *h_A, *h_mean_v, *h_sd_v, *h_t0, *h_lv, *h_uv, *h_av; 
  float *d_T0, *d_mean_w, *d_lw, *d_uw, *d_aw;
  float *h_T0, *h_mean_w, *h_sd_w, *h_lw, *h_uw, *h_aw;
  unsigned int *d_nsim, *h_nsim;
  size_t uSize  = 1 * sizeof(unsigned int);
  size_t fSize  = 1 * sizeof(float);
  size_t vfSize = nmean_v[0] * sizeof(float);
  size_t vbSize = nmean_v[0] * sizeof(bool);
  
  h_cv = (bool  *)malloc(vbSize);
  h_lv = (float *)malloc(vfSize);
  h_uv = (float *)malloc(vfSize);
  h_av = (float *)malloc(vfSize);

  h_cw = (bool  *)malloc(vbSize);
  h_lw = (float *)malloc(vfSize);
  h_uw = (float *)malloc(vfSize);
  h_aw = (float *)malloc(vfSize);

  h_nsim   = (unsigned int *)malloc(uSize);
  h_b      = (float *)malloc(fSize);
  h_A      = (float *)malloc(fSize);
  h_t0     = (float *)malloc(fSize);
  h_T0     = (float *)malloc(fSize);
  h_mean_v = (float *)malloc(vfSize);
  h_mean_w = (float *)malloc(vfSize);
  h_sd_v   = (float *)malloc(vfSize);
  h_sd_w   = (float *)malloc(vfSize);
  
  *h_nsim  = (unsigned int)*nsim;
  *h_b  = (float)*b;
  *h_A  = (float)*A;
  *h_t0 = (float)*t0;
  *h_T0 = (float)*T0;

  for(size_t i=0; i<nmean_v[0]; i++) {
    h_mean_v[i] = (float)mean_v[i];
    h_sd_v[i]   = (float)sd_v[i];
    h_lv[i] = (0 - h_mean_v[i]) / h_sd_v[i]; 
    h_uv[i] = (INFINITY - h_mean_v[i]) / h_sd_v[i]; 
    h_av[i] = 0.5 * (std::sqrt(h_lv[i]*h_lv[i] + 4.0) + h_lv[i]); 
    h_cv[i] = (h_lv[i] < 0 && h_uv[i] == INFINITY) || (h_lv[i] == -INFINITY && h_uv[i]) ||
      (isfinite(h_lv[i]) && isfinite(h_uv[i]) && h_lv[i] < 0 && h_uv[i] > 0 && 
      ((h_uv[i] - h_lv[i]) > SQRT_2PI));

    h_mean_w[i] = (float)mean_w[i];
    h_sd_w[i]   = (float)sd_v[i];
    h_lw[i] = (0 - h_mean_w[i]) / h_sd_w[i]; 
    h_uw[i] = (INFINITY - h_mean_w[i]) / h_sd_w[i]; 
    h_aw[i] = 0.5 * (std::sqrt(h_lw[i]*h_lw[i] + 4.0) + h_lw[i]); 
    h_cw[i] = (h_lw[i] < 0 && h_uw[i]==INFINITY) || (h_lw[i]==-INFINITY && h_uw[i]) ||
      (isfinite(h_lw[i]) && isfinite(h_uw[i]) && h_lw[i] < 0 && h_uw[i] > 0 && 
      ((h_uw[i] - h_lw[i]) > SQRT_2PI));
  }
  
  cudaMalloc((void**) &d_nsim, uSize);
  cudaMalloc((void**) &d_b,  fSize);
  cudaMalloc((void**) &d_A,  fSize);
  cudaMalloc((void**) &d_t0, fSize);
  cudaMalloc((void**) &d_T0, fSize);
  cudaMalloc((void**) &d_mean_v, vfSize);
  cudaMalloc((void**) &d_mean_w, vfSize);
  cudaMalloc((void**) &d_sd_v,   vfSize);
  cudaMalloc((void**) &d_lv,     vfSize);
  cudaMalloc((void**) &d_uv,     vfSize);
  cudaMalloc((void**) &d_av,     vfSize);
  cudaMalloc((void**) &d_cv,     vbSize);
  cudaMalloc((void**) &d_lw,     vfSize);
  cudaMalloc((void**) &d_uw,     vfSize);
  cudaMalloc((void**) &d_aw,     vfSize);
  cudaMalloc((void**) &d_cw,     vbSize);
  cudaMemcpy(d_nsim,   h_nsim,   uSize,  cudaMemcpyHostToDevice);
  cudaMemcpy(d_b,      h_b,      fSize,  cudaMemcpyHostToDevice);
  cudaMemcpy(d_A,      h_A,      fSize,  cudaMemcpyHostToDevice);
  cudaMemcpy(d_t0,     h_t0,     fSize,  cudaMemcpyHostToDevice);
  cudaMemcpy(d_T0,     h_T0,     fSize,  cudaMemcpyHostToDevice);
  cudaMemcpy(d_mean_v, h_mean_v, vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mean_w, h_mean_w, vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_sd_v,   h_sd_v,   vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_lv,     h_lv,    vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_uv,     h_uv,    vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_av,     h_av,    vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_cv,     h_cv,    vbSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_lw,     h_lw,    vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_uw,     h_uw,    vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_aw,     h_aw,    vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_cw,     h_cw,    vbSize, cudaMemcpyHostToDevice);
  
  rplba1_n1_kernel<<<(*nsim)/(*nth), *nth>>>(d_nsim, d_b,  d_A, d_mean_v, d_mean_w,
    d_sd_v, d_t0, d_lv, d_uv, d_av, d_cv, d_lw, d_uw, d_aw, d_cw, d_T0, d_RT, d_R);

  free(h_b);
  free(h_A);
  free(h_mean_v);
  free(h_mean_w);
  free(h_sd_v);
  free(h_t0);
  free(h_cv);
  free(h_lv);
  free(h_uv);
  free(h_av);
  free(h_cw);
  free(h_lw);
  free(h_uw);
  free(h_aw);
  free(h_nsim);
  free(h_T0);
  
  cudaFree(d_b);
  cudaFree(d_A);
  cudaFree(d_t0);
  cudaFree(d_mean_v);
  cudaFree(d_mean_w);
  cudaFree(d_sd_v);
  cudaFree(d_lv);
  cudaFree(d_uv);
  cudaFree(d_av);
  cudaFree(d_cv);
  
  cudaFree(d_lw);
  cudaFree(d_uw);
  cudaFree(d_aw);
  cudaFree(d_cw);
  
  cudaFree(d_nsim);
  cudaFree(d_T0);
}

void rplba2_n1(int *nsim, double *b, double *A, double *mean_v, int *nmean_v, 
  double *mean_w, double *sd_v, double *sd_w,  double *t0, double *T0, int *nth, unsigned int *d_R,
  float *d_RT)
{
  bool *h_cv, *d_cv, *h_cw, *d_cw; 
  float *d_b, *d_A, *d_mean_v, *d_sd_v, *d_t0, *d_lv, *d_uv, *d_av;
  float *h_b, *h_A, *h_mean_v, *h_sd_v, *h_t0, *h_lv, *h_uv, *h_av; 
  float *d_T0, *d_mean_w, *d_sd_w, *d_lw, *d_uw, *d_aw;
  float *h_T0, *h_mean_w, *h_sd_w, *h_lw, *h_uw, *h_aw;
  unsigned int *d_nsim, *h_nsim;
  size_t uSize  = 1 * sizeof(unsigned int);
  size_t fSize  = 1 * sizeof(float);
  size_t vfSize = nmean_v[0] * sizeof(float);
  size_t vbSize = nmean_v[0] * sizeof(bool);
  
  h_cv = (bool  *)malloc(vbSize);
  h_lv = (float *)malloc(vfSize);
  h_uv = (float *)malloc(vfSize);
  h_av = (float *)malloc(vfSize);

  h_cw = (bool  *)malloc(vbSize);
  h_lw = (float *)malloc(vfSize);
  h_uw = (float *)malloc(vfSize);
  h_aw = (float *)malloc(vfSize);

  h_nsim   = (unsigned int *)malloc(uSize);
  h_b      = (float *)malloc(vfSize);
  h_A      = (float *)malloc(vfSize);
  h_t0     = (float *)malloc(fSize);
  h_T0     = (float *)malloc(fSize);
  h_mean_v = (float *)malloc(vfSize);
  h_mean_w = (float *)malloc(vfSize);
  h_sd_v   = (float *)malloc(vfSize);
  h_sd_w   = (float *)malloc(vfSize);
  
  *h_nsim  = (unsigned int)*nsim;
  //*h_b  = (float)*b;
  //*h_A  = (float)*A;
  *h_t0 = (float)*t0;
  *h_T0 = (float)*T0;

  for(size_t i=0; i<nmean_v[0]; i++) {
      h_A[i] = (float)A[i];
      h_b[i] = (float)b[i];

    h_mean_v[i] = (float)mean_v[i];
    h_sd_v[i]   = (float)sd_v[i];
    h_lv[i] = (0 - h_mean_v[i]) / h_sd_v[i]; 
    h_uv[i] = (INFINITY - h_mean_v[i]) / h_sd_v[i]; 
    h_av[i] = 0.5 * (std::sqrt(h_lv[i]*h_lv[i] + 4.0) + h_lv[i]); 
    h_cv[i] = (h_lv[i] < 0 && h_uv[i] == INFINITY) || (h_lv[i] == -INFINITY && h_uv[i]) ||
      (isfinite(h_lv[i]) && isfinite(h_uv[i]) && h_lv[i] < 0 && h_uv[i] > 0 && 
      ((h_uv[i] - h_lv[i]) > SQRT_2PI));

    h_mean_w[i] = (float)mean_w[i];
    h_sd_w[i]   = (float)sd_w[i];
    h_lw[i] = (0 - h_mean_w[i]) / h_sd_w[i]; 
    h_uw[i] = (INFINITY - h_mean_w[i]) / h_sd_w[i]; 
    h_aw[i] = 0.5 * (std::sqrt(h_lw[i]*h_lw[i] + 4.0) + h_lw[i]); 
    h_cw[i] = (h_lw[i] < 0 && h_uw[i]==INFINITY) || (h_lw[i]==-INFINITY && h_uw[i]) ||
      (isfinite(h_lw[i]) && isfinite(h_uw[i]) && h_lw[i] < 0 && h_uw[i] > 0 && 
      ((h_uw[i] - h_lw[i]) > SQRT_2PI));
  }
  
  cudaMalloc((void**) &d_nsim, uSize);
  cudaMalloc((void**) &d_t0, fSize);
  cudaMalloc((void**) &d_T0, fSize);
  cudaMalloc((void**) &d_b,      vfSize);
  cudaMalloc((void**) &d_A,      vfSize);
  cudaMalloc((void**) &d_mean_v, vfSize);
  cudaMalloc((void**) &d_mean_w, vfSize);
  cudaMalloc((void**) &d_sd_v,   vfSize);
  cudaMalloc((void**) &d_sd_w,   vfSize);
  cudaMalloc((void**) &d_lv,     vfSize);
  cudaMalloc((void**) &d_uv,     vfSize);
  cudaMalloc((void**) &d_av,     vfSize);
  cudaMalloc((void**) &d_cv,     vbSize);
  cudaMalloc((void**) &d_lw,     vfSize);
  cudaMalloc((void**) &d_uw,     vfSize);
  cudaMalloc((void**) &d_aw,     vfSize);
  cudaMalloc((void**) &d_cw,     vbSize);
  cudaMemcpy(d_nsim,   h_nsim,   uSize,  cudaMemcpyHostToDevice);
  cudaMemcpy(d_b,      h_b,      vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_A,      h_A,      vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_t0,     h_t0,     fSize,  cudaMemcpyHostToDevice);
  cudaMemcpy(d_T0,     h_T0,     fSize,  cudaMemcpyHostToDevice);
  cudaMemcpy(d_mean_v, h_mean_v, vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mean_w, h_mean_w, vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_sd_v,   h_sd_v,  vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_sd_w,   h_sd_w,  vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_lv,     h_lv,    vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_uv,     h_uv,    vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_av,     h_av,    vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_cv,     h_cv,    vbSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_lw,     h_lw,    vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_uw,     h_uw,    vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_aw,     h_aw,    vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_cw,     h_cw,    vbSize, cudaMemcpyHostToDevice);
  
  rplba2_n1_kernel<<<(*nsim)/(*nth), *nth>>>(d_nsim, d_b,  d_A, d_mean_v, d_mean_w,
                                             d_sd_v, d_sd_w, d_t0, d_lv, d_uv, d_av,
                                             d_cv, d_lw, d_uw, d_aw, d_cw, d_T0, d_RT, d_R);

  free(h_b);
  free(h_A);
  free(h_mean_v);
  free(h_mean_w);
  free(h_sd_v);
  free(h_t0);
  free(h_cv);
  free(h_lv);
  free(h_uv);
  free(h_av);
  free(h_cw);
  free(h_lw);
  free(h_uw);
  free(h_aw);
  free(h_nsim);
  free(h_T0);
  
  cudaFree(d_b);
  cudaFree(d_A);
  cudaFree(d_t0);
  cudaFree(d_mean_v);
  cudaFree(d_mean_w);
  cudaFree(d_sd_v);
  cudaFree(d_lv);
  cudaFree(d_uv);
  cudaFree(d_av);
  cudaFree(d_cv);
  
  cudaFree(d_lw);
  cudaFree(d_uw);
  cudaFree(d_aw);
  cudaFree(d_cw);
  
  cudaFree(d_nsim);
  cudaFree(d_T0);
}

void rplba2_entry(int *nsim, double *b, double *A, double *mean_v, int *nmean_v, 
  double *mean_w, double *sd_v, double *sd_w,  double *t0, double *T0, int *nth, int *R,
  double *RT)
{
  bool *h_cv, *d_cv, *h_cw, *d_cw; 
  float *d_b, *d_A, *d_mean_v, *d_sd_v, *d_t0, *d_lv, *d_uv, *d_av;
  float *h_b, *h_A, *h_mean_v, *h_sd_v, *h_t0, *h_lv, *h_uv, *h_av; 
  float *d_T0, *d_mean_w, *d_sd_w, *d_lw, *d_uw, *d_aw;
  float *h_T0, *h_mean_w, *h_sd_w, *h_lw, *h_uw, *h_aw;
  unsigned int *d_nsim, *h_nsim;
  size_t uSize  = 1 * sizeof(unsigned int);
  size_t fSize  = 1 * sizeof(float);
  size_t vfSize = nmean_v[0] * sizeof(float);
  size_t vbSize = nmean_v[0] * sizeof(bool);
  size_t nuSize  = *nsim * sizeof(unsigned int);
  size_t nfSize  = *nsim * sizeof(float);

  float *h_RT, *d_RT;
  unsigned int *h_R, *d_R;

  h_cv = (bool  *)malloc(vbSize);
  h_lv = (float *)malloc(vfSize);
  h_uv = (float *)malloc(vfSize);
  h_av = (float *)malloc(vfSize);

  h_cw = (bool  *)malloc(vbSize);
  h_lw = (float *)malloc(vfSize);
  h_uw = (float *)malloc(vfSize);
  h_aw = (float *)malloc(vfSize);

  h_nsim   = (unsigned int *)malloc(uSize);
  h_b      = (float *)malloc(vfSize);
  h_A      = (float *)malloc(vfSize);
  h_t0     = (float *)malloc(fSize);
  h_T0     = (float *)malloc(fSize);
  h_mean_v = (float *)malloc(vfSize);
  h_mean_w = (float *)malloc(vfSize);
  h_sd_v   = (float *)malloc(vfSize);
  h_sd_w   = (float *)malloc(vfSize);
  
  *h_nsim  = (unsigned int)*nsim;
  *h_t0 = (float)*t0;
  *h_T0 = (float)*T0;

  for(size_t i=0; i<nmean_v[0]; i++) {
      h_A[i] = (float)A[i];
      h_b[i] = (float)b[i];

    h_mean_v[i] = (float)mean_v[i];
    h_sd_v[i]   = (float)sd_v[i];
    h_lv[i] = (0 - h_mean_v[i]) / h_sd_v[i]; 
    h_uv[i] = (INFINITY - h_mean_v[i]) / h_sd_v[i]; 
    h_av[i] = 0.5 * (std::sqrt(h_lv[i]*h_lv[i] + 4.0) + h_lv[i]); 
    h_cv[i] = (h_lv[i] < 0 && h_uv[i] == INFINITY) || (h_lv[i] == -INFINITY && h_uv[i]) ||
      (isfinite(h_lv[i]) && isfinite(h_uv[i]) && h_lv[i] < 0 && h_uv[i] > 0 && 
      ((h_uv[i] - h_lv[i]) > SQRT_2PI));

    h_mean_w[i] = (float)mean_w[i];
    h_sd_w[i]   = (float)sd_w[i];
    h_lw[i] = (0 - h_mean_w[i]) / h_sd_w[i]; 
    h_uw[i] = (INFINITY - h_mean_w[i]) / h_sd_w[i]; 
    h_aw[i] = 0.5 * (std::sqrt(h_lw[i]*h_lw[i] + 4.0) + h_lw[i]); 
    h_cw[i] = (h_lw[i] < 0 && h_uw[i]==INFINITY) || (h_lw[i]==-INFINITY && h_uw[i]) ||
      (isfinite(h_lw[i]) && isfinite(h_uw[i]) && h_lw[i] < 0 && h_uw[i] > 0 && 
      ((h_uw[i] - h_lw[i]) > SQRT_2PI));
  }

  cudaHostAlloc((void**)&h_RT,   nfSize, cudaHostAllocDefault);
  cudaHostAlloc((void**)&h_R,    nuSize, cudaHostAllocDefault);
  cudaMalloc((void**) &d_RT, nuSize);
  cudaMalloc((void**) &d_R,  nfSize);

  cudaMalloc((void**) &d_nsim, uSize);
  cudaMalloc((void**) &d_t0, fSize);
  cudaMalloc((void**) &d_T0, fSize);
  cudaMalloc((void**) &d_b,      vfSize);
  cudaMalloc((void**) &d_A,      vfSize);
  cudaMalloc((void**) &d_mean_v, vfSize);
  cudaMalloc((void**) &d_mean_w, vfSize);
  cudaMalloc((void**) &d_sd_v,   vfSize);
  cudaMalloc((void**) &d_sd_w,   vfSize);
  cudaMalloc((void**) &d_lv,     vfSize);
  cudaMalloc((void**) &d_uv,     vfSize);
  cudaMalloc((void**) &d_av,     vfSize);
  cudaMalloc((void**) &d_cv,     vbSize);
  cudaMalloc((void**) &d_lw,     vfSize);
  cudaMalloc((void**) &d_uw,     vfSize);
  cudaMalloc((void**) &d_aw,     vfSize);
  cudaMalloc((void**) &d_cw,     vbSize);
  cudaMemcpy(d_nsim,   h_nsim,   uSize,  cudaMemcpyHostToDevice);
  cudaMemcpy(d_b,      h_b,      vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_A,      h_A,      vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_t0,     h_t0,     fSize,  cudaMemcpyHostToDevice);
  cudaMemcpy(d_T0,     h_T0,     fSize,  cudaMemcpyHostToDevice);
  cudaMemcpy(d_mean_v, h_mean_v, vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mean_w, h_mean_w, vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_sd_v,   h_sd_v,  vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_sd_w,   h_sd_w,  vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_lv,     h_lv,    vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_uv,     h_uv,    vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_av,     h_av,    vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_cv,     h_cv,    vbSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_lw,     h_lw,    vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_uw,     h_uw,    vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_aw,     h_aw,    vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_cw,     h_cw,    vbSize, cudaMemcpyHostToDevice);
  
  rplba2_kernel<<<(*nsim)/(*nth), *nth>>>(d_nsim, d_b,  d_A, d_mean_v, d_mean_w,
                                             d_sd_v, d_sd_w, d_t0, d_lv, d_uv, d_av,
                                             d_cv, d_lw, d_uw, d_aw, d_cw, d_T0, d_RT, d_R);

  cudaMemcpy(h_RT, d_RT, nfSize, cudaMemcpyDeviceToHost); cudaFree(d_RT);
  cudaMemcpy(h_R,  d_R,  nuSize, cudaMemcpyDeviceToHost); cudaFree(d_R);
  for(size_t i=0; i<*nsim; i++) {
      RT[i] = (double)h_RT[i];
      R[i]  = (int)h_R[i];
  }
  cudaFreeHost(h_RT);
  cudaFreeHost(h_R);

  free(h_b);
  free(h_A);
  free(h_mean_v);
  free(h_mean_w);
  free(h_sd_v);
  free(h_t0);
  free(h_cv);
  free(h_lv);
  free(h_uv);
  free(h_av);
  free(h_cw);
  free(h_lw);
  free(h_uw);
  free(h_aw);
  free(h_nsim);
  free(h_T0);
  
  cudaFree(d_b);
  cudaFree(d_A);
  cudaFree(d_t0);
  cudaFree(d_mean_v);
  cudaFree(d_mean_w);
  cudaFree(d_sd_v);
  cudaFree(d_lv);
  cudaFree(d_uv);
  cudaFree(d_av);
  cudaFree(d_cv);
  
  cudaFree(d_lw);
  cudaFree(d_uw);
  cudaFree(d_aw);
  cudaFree(d_cw);
  
  cudaFree(d_nsim);
  cudaFree(d_T0);
}


void rplba3_n1(int *nsim, float *b, double *A, float* c, double *mean_v,
               int *nmean_v, double *mean_w, double *sd_v, double *sd_w,
               double *t0,
               float *swt1, float *swt2, float *swtD, bool *a, int *nth,
               unsigned int *d_R, float *d_RT)
{
  bool *d_a;
  float *d_b, *d_c, *d_swt1, *d_swt2, *d_swtD;

  bool *h_cv, *d_cv, *h_cw, *d_cw;
  float *d_A, *d_mean_v, *d_sd_v, *d_t0, *d_lv, *d_uv, *d_av;
  float *h_A, *h_mean_v, *h_sd_v, *h_t0, *h_lv, *h_uv, *h_av; 
  float *d_mean_w, *d_sd_w, *d_lw, *d_uw, *d_aw;
  float *h_mean_w, *h_sd_w, *h_lw, *h_uw, *h_aw;
  unsigned int *d_nsim, *h_nsim;
  size_t uSize  = 1 * sizeof(unsigned int);
  size_t fSize  = 1 * sizeof(float);
  size_t vfSize = nmean_v[0] * sizeof(float);
  size_t vbSize = nmean_v[0] * sizeof(bool);
  
  h_cv = (bool  *)malloc(vbSize);
  h_lv = (float *)malloc(vfSize);
  h_uv = (float *)malloc(vfSize);
  h_av = (float *)malloc(vfSize);

  h_cw = (bool  *)malloc(vbSize);
  h_lw = (float *)malloc(vfSize);
  h_uw = (float *)malloc(vfSize);
  h_aw = (float *)malloc(vfSize);

  h_nsim   = (unsigned int *)malloc(uSize);
  h_A      = (float *)malloc(vfSize);
  h_t0     = (float *)malloc(fSize);
  h_mean_v = (float *)malloc(vfSize);
  h_mean_w = (float *)malloc(vfSize);
  h_sd_v   = (float *)malloc(vfSize);
  h_sd_w   = (float *)malloc(vfSize);
  *h_nsim  = (unsigned int)*nsim;

  *h_t0 = (float)*t0;

  for(size_t i=0; i<nmean_v[0]; i++) {
    h_A[i] = (float)A[i];
    h_mean_v[i] = (float)mean_v[i];
    h_sd_v[i]   = (float)sd_v[i];
    h_lv[i] = (0 - h_mean_v[i]) / h_sd_v[i]; 
    h_uv[i] = (INFINITY - h_mean_v[i]) / h_sd_v[i]; 
    h_av[i] = 0.5 * (std::sqrt(h_lv[i]*h_lv[i] + 4.0) + h_lv[i]); 
    h_cv[i] = (h_lv[i] < 0 && h_uv[i] == INFINITY) || (h_lv[i] == -INFINITY && h_uv[i]) ||
      (isfinite(h_lv[i]) && isfinite(h_uv[i]) && h_lv[i] < 0 && h_uv[i] > 0 && 
      ((h_uv[i] - h_lv[i]) > SQRT_2PI));

    h_mean_w[i] = (float)mean_w[i];
    h_sd_w[i]   = (float)sd_w[i];
    h_lw[i] = (0 - h_mean_w[i]) / h_sd_w[i]; 
    h_uw[i] = (INFINITY - h_mean_w[i]) / h_sd_w[i]; 
    h_aw[i] = 0.5 * (std::sqrt(h_lw[i]*h_lw[i] + 4.0) + h_lw[i]); 
    h_cw[i] = (h_lw[i] < 0 && h_uw[i]==INFINITY) || (h_lw[i]==-INFINITY && h_uw[i]) ||
      (isfinite(h_lw[i]) && isfinite(h_uw[i]) && h_lw[i] < 0 && h_uw[i] > 0 && 
      ((h_uw[i] - h_lw[i]) > SQRT_2PI));
  }
  
  cudaMalloc((void**) &d_nsim, uSize);
  cudaMalloc((void**) &d_t0,   fSize);
  cudaMalloc((void**) &d_swt1, fSize);
  cudaMalloc((void**) &d_swt2, fSize);
  cudaMalloc((void**) &d_swtD, fSize);
  cudaMalloc((void**) &d_b,      vfSize);
  cudaMalloc((void**) &d_A,      vfSize);
  cudaMalloc((void**) &d_c,      vfSize);
  cudaMalloc((void**) &d_a,      sizeof(bool) * 3);
  cudaMalloc((void**) &d_mean_v, vfSize);
  cudaMalloc((void**) &d_mean_w, vfSize);
  cudaMalloc((void**) &d_sd_v,   vfSize);
  cudaMalloc((void**) &d_sd_w,   vfSize);
  cudaMalloc((void**) &d_lv,     vfSize);
  cudaMalloc((void**) &d_uv,     vfSize);
  cudaMalloc((void**) &d_av,     vfSize);
  cudaMalloc((void**) &d_cv,     vbSize);
  cudaMalloc((void**) &d_lw,     vfSize);
  cudaMalloc((void**) &d_uw,     vfSize);
  cudaMalloc((void**) &d_aw,     vfSize);
  cudaMalloc((void**) &d_cw,     vbSize);
  cudaMemcpy(d_nsim,   h_nsim,   uSize,  cudaMemcpyHostToDevice);
  cudaMemcpy(d_b,      b,        vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_A,      h_A,      vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c,      c,        vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_a,      a,        sizeof(bool)*3, cudaMemcpyHostToDevice);
  cudaMemcpy(d_t0,     h_t0,     fSize,  cudaMemcpyHostToDevice);
  cudaMemcpy(d_swt1,   swt1,     fSize,  cudaMemcpyHostToDevice);
  cudaMemcpy(d_swt2,   swt2,     fSize,  cudaMemcpyHostToDevice);
  cudaMemcpy(d_swtD,   swtD,     fSize,  cudaMemcpyHostToDevice);
  cudaMemcpy(d_mean_v, h_mean_v, vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mean_w, h_mean_w, vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_sd_v,   h_sd_v,   vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_sd_w,   h_sd_w,   vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_lv,     h_lv,     vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_uv,     h_uv,     vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_av,     h_av,     vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_cv,     h_cv,     vbSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_lw,     h_lw,     vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_uw,     h_uw,     vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_aw,     h_aw,     vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_cw,     h_cw,     vbSize, cudaMemcpyHostToDevice);

  rplba3_n1_kernel<<<(*nsim)/(*nth), *nth>>>(d_nsim, d_b, d_A, d_c, d_a, d_mean_v, d_mean_w,
                                             d_sd_v, d_sd_w, d_t0, d_lv, d_uv, d_av,
                                             d_cv, d_lw, d_uw, d_aw, d_cw, d_swt1, d_swt2,
                                             d_swtD, d_RT, d_R);

  free(h_A);
  free(h_mean_v);
  free(h_mean_w);
  free(h_sd_v);
  free(h_t0);
  free(h_cv);
  free(h_lv);
  free(h_uv);
  free(h_av);
  free(h_cw);
  free(h_lw);
  free(h_uw);
  free(h_aw);
  free(h_nsim);
  
  cudaFree(d_b);
  cudaFree(d_A);
  cudaFree(d_c);
  cudaFree(d_a);
  cudaFree(d_t0);
  cudaFree(d_mean_v);
  cudaFree(d_mean_w);
  cudaFree(d_sd_v);
  cudaFree(d_lv);
  cudaFree(d_uv);
  cudaFree(d_av);
  cudaFree(d_cv);
  
  cudaFree(d_lw);
  cudaFree(d_uw);
  cudaFree(d_aw);
  cudaFree(d_cw);
  
  cudaFree(d_nsim);
  cudaFree(d_swt1);
  cudaFree(d_swt2);
  cudaFree(d_swtD);
}


void rplba3_entry(int *nsim, double *b, double *A, double* c, double *mean_v,
               int *nmean_v, double *mean_w, double *sd_v, double *sd_w,
               double *t0, double *swt1, double *swt2, double *swtD, bool *a, int *nth,
               int *R, double *RT)
{
  bool *d_a;
  float *d_b, *d_c, *d_swt1, *d_swt2, *d_swtD;
  float *h_b, *h_c, *h_swt1, *h_swt2, *h_swtD;

  bool *h_cv, *d_cv, *h_cw, *d_cw;
  float *d_A, *d_mean_v, *d_sd_v, *d_t0, *d_lv, *d_uv, *d_av;
  float *h_A, *h_mean_v, *h_sd_v, *h_t0, *h_lv, *h_uv, *h_av; 
  float *d_mean_w, *d_sd_w, *d_lw, *d_uw, *d_aw;
  float *h_mean_w, *h_sd_w, *h_lw, *h_uw, *h_aw;
  unsigned int *d_nsim, *h_nsim;
  size_t uSize  = 1 * sizeof(unsigned int);
  size_t fSize  = 1 * sizeof(float);
  size_t vfSize = nmean_v[0] * sizeof(float);
  size_t vbSize = nmean_v[0] * sizeof(bool);
  size_t nuSize  = *nsim * sizeof(unsigned int);
  size_t nfSize  = *nsim * sizeof(float);

  float *h_RT, *d_RT;
  unsigned int *h_R, *d_R;
  
  h_cv = (bool  *)malloc(vbSize);
  h_lv = (float *)malloc(vfSize);
  h_uv = (float *)malloc(vfSize);
  h_av = (float *)malloc(vfSize);

  h_cw = (bool  *)malloc(vbSize);
  h_lw = (float *)malloc(vfSize);
  h_uw = (float *)malloc(vfSize);
  h_aw = (float *)malloc(vfSize);

  h_nsim   = (unsigned int *)malloc(uSize);
  h_A      = (float *)malloc(vfSize);
  h_t0     = (float *)malloc(fSize);
  h_mean_v = (float *)malloc(vfSize);
  h_mean_w = (float *)malloc(vfSize);
  h_sd_v   = (float *)malloc(vfSize);
  h_sd_w   = (float *)malloc(vfSize);
  *h_nsim  = (unsigned int)*nsim;

  h_b   = (float *)malloc(vfSize);
  h_c   = (float *)malloc(vfSize);
  h_swt1= (float *)malloc(fSize);
  h_swt2= (float *)malloc(fSize);
  h_swtD= (float *)malloc(fSize);

  *h_t0 = (float)*t0;
  *h_swt1 = (float)*swt1;
  *h_swt2 = (float)*swt2;
  *h_swtD = (float)*swtD;

  for(size_t i=0; i<nmean_v[0]; i++) {
    h_A[i] = (float)A[i];
    h_b[i] = (float)b[i];
    h_c[i] = (float)c[i];

    h_mean_v[i] = (float)mean_v[i];
    h_sd_v[i]   = (float)sd_v[i];
    h_lv[i] = (0 - h_mean_v[i]) / h_sd_v[i]; 
    h_uv[i] = (INFINITY - h_mean_v[i]) / h_sd_v[i]; 
    h_av[i] = 0.5 * (std::sqrt(h_lv[i]*h_lv[i] + 4.0) + h_lv[i]); 
    h_cv[i] = (h_lv[i] < 0 && h_uv[i] == INFINITY) || (h_lv[i] == -INFINITY && h_uv[i]) ||
      (isfinite(h_lv[i]) && isfinite(h_uv[i]) && h_lv[i] < 0 && h_uv[i] > 0 && 
      ((h_uv[i] - h_lv[i]) > SQRT_2PI));

    h_mean_w[i] = (float)mean_w[i];
    h_sd_w[i]   = (float)sd_w[i];
    h_lw[i] = (0 - h_mean_w[i]) / h_sd_w[i]; 
    h_uw[i] = (INFINITY - h_mean_w[i]) / h_sd_w[i]; 
    h_aw[i] = 0.5 * (std::sqrt(h_lw[i]*h_lw[i] + 4.0) + h_lw[i]); 
    h_cw[i] = (h_lw[i] < 0 && h_uw[i]==INFINITY) || (h_lw[i]==-INFINITY && h_uw[i]) ||
      (isfinite(h_lw[i]) && isfinite(h_uw[i]) && h_lw[i] < 0 && h_uw[i] > 0 && 
      ((h_uw[i] - h_lw[i]) > SQRT_2PI));
  }

  cudaHostAlloc((void**)&h_RT,   nfSize, cudaHostAllocDefault);
  cudaHostAlloc((void**)&h_R,    nuSize, cudaHostAllocDefault);
  cudaMalloc((void**) &d_RT, nuSize);
  cudaMalloc((void**) &d_R,  nfSize);

  cudaMalloc((void**) &d_nsim, uSize);
  cudaMalloc((void**) &d_t0,   fSize);
  cudaMalloc((void**) &d_swt1, fSize);
  cudaMalloc((void**) &d_swt2, fSize);
  cudaMalloc((void**) &d_swtD, fSize);
  cudaMalloc((void**) &d_b,      vfSize);
  cudaMalloc((void**) &d_A,      vfSize);
  cudaMalloc((void**) &d_c,      vfSize);
  cudaMalloc((void**) &d_a,      sizeof(bool) * 3);
  cudaMalloc((void**) &d_mean_v, vfSize);
  cudaMalloc((void**) &d_mean_w, vfSize);
  cudaMalloc((void**) &d_sd_v,   vfSize);
  cudaMalloc((void**) &d_sd_w,   vfSize);
  cudaMalloc((void**) &d_lv,     vfSize);
  cudaMalloc((void**) &d_uv,     vfSize);
  cudaMalloc((void**) &d_av,     vfSize);
  cudaMalloc((void**) &d_cv,     vbSize);
  cudaMalloc((void**) &d_lw,     vfSize);
  cudaMalloc((void**) &d_uw,     vfSize);
  cudaMalloc((void**) &d_aw,     vfSize);
  cudaMalloc((void**) &d_cw,     vbSize);
  cudaMemcpy(d_nsim,   h_nsim,   uSize,  cudaMemcpyHostToDevice);
  cudaMemcpy(d_b,      h_b,      vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_A,      h_A,      vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c,      h_c,      vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_a,      a,        sizeof(bool)*3, cudaMemcpyHostToDevice);
  cudaMemcpy(d_t0,     h_t0,     fSize,  cudaMemcpyHostToDevice);
  cudaMemcpy(d_swt1,   h_swt1,   fSize,  cudaMemcpyHostToDevice);
  cudaMemcpy(d_swt2,   h_swt2,   fSize,  cudaMemcpyHostToDevice);
  cudaMemcpy(d_swtD,   h_swtD,   fSize,  cudaMemcpyHostToDevice);
  cudaMemcpy(d_mean_v, h_mean_v, vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mean_w, h_mean_w, vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_sd_v,   h_sd_v,   vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_sd_w,   h_sd_w,   vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_lv,     h_lv,     vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_uv,     h_uv,     vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_av,     h_av,     vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_cv,     h_cv,     vbSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_lw,     h_lw,     vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_uw,     h_uw,     vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_aw,     h_aw,     vfSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_cw,     h_cw,     vbSize, cudaMemcpyHostToDevice);

  rplba3_kernel<<<(*nsim)/(*nth), *nth>>>(d_nsim, d_b, d_A, d_c, d_a, d_mean_v, d_mean_w,
                                             d_sd_v, d_sd_w, d_t0, d_lv, d_uv, d_av,
                                             d_cv, d_lw, d_uw, d_aw, d_cw, d_swt1, d_swt2,
                                             d_swtD, d_RT, d_R);

  cudaMemcpy(h_RT, d_RT, nfSize, cudaMemcpyDeviceToHost); cudaFree(d_RT);
  cudaMemcpy(h_R,  d_R,  nuSize, cudaMemcpyDeviceToHost); cudaFree(d_R);
  for(size_t i=0; i<*nsim; i++) {
      RT[i] = (double)h_RT[i];
      R[i]  = (int)h_R[i];
  }
  cudaFreeHost(h_RT);
  cudaFreeHost(h_R);

  free(h_A);
  free(h_mean_v);
  free(h_mean_w);
  free(h_sd_v);
  free(h_t0);
  free(h_cv);
  free(h_lv);
  free(h_uv);
  free(h_av);
  free(h_cw);
  free(h_lw);
  free(h_uw);
  free(h_aw);
  free(h_nsim);
  
  cudaFree(d_b);
  cudaFree(d_A);
  cudaFree(d_c);
  cudaFree(d_a);
  cudaFree(d_t0);
  cudaFree(d_mean_v);
  cudaFree(d_mean_w);
  cudaFree(d_sd_v);
  cudaFree(d_lv);
  cudaFree(d_uv);
  cudaFree(d_av);
  cudaFree(d_cv);
  
  cudaFree(d_lw);
  cudaFree(d_uw);
  cudaFree(d_aw);
  cudaFree(d_cw);
  
  cudaFree(d_nsim);
  cudaFree(d_swt1);
  cudaFree(d_swt2);
  cudaFree(d_swtD);
}
