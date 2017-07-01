// CUDA kernel functions to generate random numbers from the uniform 
// distribution on the interval from min to max
__global__ void runif_kernel(int n, double* min, double* max, double* out);
__global__ void runif_kernel(int n, float* min, float* max, float* out);

// CUDA kernel functions to generate random numbers from the normal 
// distribution with mean eqaul to mean and standard deviation to sd
__global__ void rnorm_kernel(int n, double* mean, double* sd, double* out);
__global__ void rnorm_kernel(int n, float* mean, float* sd, float* out);

// CUDA kernel functions to generate random numbers from the truncated normal 
// distribution with mean eqaul to mean, standard deviation to sd, lower bound
// equal to l, and upper bound equal to u. rtnorm[0-3] stands for algorithm
// 1, 2, 3 and 4 (see Robert, 1995).
__global__ void rtnorm0_kernel(int n, double* mean, double* sd, double* l,
                               double* u, double* out);
__global__ void rtnorm1_kernel(int n, double* mean, double* sd, double* l,
                               double* u, double* out);
__global__ void rtnorm2_kernel(int n, double* mean, double* sd, double* l,
                               double* u, double* out);
__global__ void rtnorm3_kernel(int n, double* mean, double* sd, double* l,
                               double* u, double* out);
__global__ void rtnorm0_kernel(int n, float* mean, float* sd, float* l,
                               float* u, float* out);
__global__ void rtnorm1_kernel(int n, float* mean, float* sd, float* l,
                               float* u, float* out) ;
__global__ void rtnorm2_kernel(int n, float* mean, float* sd, float* l,
                               float* u, float* out);
__global__ void rtnorm3_kernel(int n, float* mean, float* sd, float* l,
                               float* u, float* out) ;

// CUDA device functions to generate random numbers from the truncated normal
// distribution: algorithm 1 and 2
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

// CUDA kernel functions to generate random numbers from the canonical LBA 
// model with parameters: b, A, mean_v, sd_v and t0. The four inputs, lower, 
// upper, a and c, are for rtnorm[0,1]_device to draw trial drift rates from 
// the truncated normal disbribution. a stands fro alpha star in Robert (1995).
// (see Brown & Heathcote, 2008 for canonical LBA model). RT and R are the 
// outputs, standing respectively for response times (in second) and responses
// (1 or 2 for choice 1 or choice 2)
__global__ void rlba_kernel(int* n, double* b, double* A, double* mean_v,
  double* sd_v, double* t0, double* lower,
  double* upper, double* a, bool* c, double* RT, unsigned int* R);
__global__ void rlba_kernel(int* n, float* b, float* A, float* mean_v,
                            float* sd_v, float* t0, float* lower, float* upper,
                            float* a, bool* c, float* RT, unsigned int* R);

// CUDA kernel functions to generate random numbers from the canonical LBA 
// model. This function return only node 1 (ie choice 1) outputs.
__global__ void rlba_n1_kernel(unsigned int *n, double *b, double *A,
                               double *mean_v, double *sd_v, double *t0,
                               double *lower, double *upper, double *a,
                               bool *c, double *RT, unsigned int *R);
__global__ void rlba_n1_kernel(unsigned int *n, float *b, float *A,
                               float *mean_v, float *sd_v, float *t0,
                               float *lower, float *upper, float *a,
                               bool *c, float *RT, unsigned int *R);

// A C function to access rlba_n1_kernel  
void rn1(int *nsim, double *b, double *A, double *mean_v, int *nmean_v,
  double *sd_v, double *t0, int *nth, unsigned int *d_R, float *d_RT0);


// A C function to access rplba1_n1_kernel. rplba[1-3] stands for three 
// types of PLBA models, each with a different combination of rate and 
// threshold delays
void rplba1_n1(int *nsim, double *b, double *A, double *mean_v, int *nmean_v,
              double *mean_w, double *sd_v, double *t0, double *T0, int *nth,
              unsigned int *d_R, float *d_RT);

// A C function to access rplba2_n1_kernel  
void rplba2_n1(int *nsim, double *b, double *A, double *mean_v, int *nmean_v,
               double *mean_w, double *sd_v, double *sd_w,  double *t0,
               double *T0, int *nth, unsigned int *d_R, float *d_RT);

// A C function to access rplba3_n1_kernel  
void rplba3_n1(int *nsim, float *b, double *A, float* c, double *mean_v,
               int *nmean_v, double *mean_w, double *sd_v, double *sd_w,
               double *t0, float *swt1, float *swt2, float *swtD, bool *a,
               int *nth, unsigned int *d_R, float *d_RT);
