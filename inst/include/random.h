__global__ void runif_kernel(int n, double* min, double* max, double* out);
__global__ void runif_kernel(int n, float* min, float* max, float* out);

__global__ void rnorm_kernel(int n, double* mean, double* sd, double* out);
__global__ void rnorm_kernel(int n, float* mean, float* sd, float* out);

__global__ void rtnorm0_kernel(int n, double* mean, double* sd, double* l,
  double* u, double* out);
__global__ void rtnorm1_kernel(int n, double* mean, double* sd, double* l,
  double* u, double* out) ;
__global__ void rtnorm2_kernel(int n, double* mean, double* sd, double* l,
  double* u, double* out);
__global__ void rtnorm3_kernel(int n, double* mean, double* sd, double* l,
  double* u, double* out) ;

__global__ void rtnorm0_kernel(int n, float* mean, float* sd, float* l,
                               float* u, float* out);
__global__ void rtnorm1_kernel(int n, float* mean, float* sd, float* l,
                               float* u, float* out) ;
__global__ void rtnorm2_kernel(int n, float* mean, float* sd, float* l,
                               float* u, float* out);
__global__ void rtnorm3_kernel(int n, float* mean, float* sd, float* l,
                               float* u, float* out) ;

static __device__ inline 
  double rtnorm0_device(curandState_t* state, double mean, double sd,
                        double l, double u);
static __device__ inline
  float rtnorm0_device(curandState_t* state, float mean, float sd,
                       float l, float u);
static __device__ inline
  double rtnorm1_device(curandState_t* state, double a, double mean,
                        double sd, double l, double u);
static __device__ inline
  float rtnorm1_device(curandState_t* state, float a, float mean,
                       float sd, float l, float u);


__global__ void rlba_kernel(int* n, double* b, double* A, double* mean_v,
  double* sd_v, double* t0, double* lower,
  double* upper, double* a, bool* c, double* RT, unsigned int* R);
__global__ void rlba_kernel(int* n, float* b, float* A, float* mean_v,
                            float* sd_v, float* t0, float* lower, float* upper,
                            float* a, bool* c, float* RT, unsigned int* R);

__global__ void rlba_n1_kernel(unsigned int *n, double *b, double *A,
                               double *mean_v, double *sd_v, double *t0,
                               double *lower, double *upper, double *a,
                               bool *c, double *RT, unsigned int *R);
__global__ void rlba_n1_kernel(unsigned int *n, float *b, float *A,
                               float *mean_v, float *sd_v, float *t0,
                               float *lower, float *upper, float *a,
                               bool *c, float *RT, unsigned int *R);

