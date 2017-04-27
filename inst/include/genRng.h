__global__ void runif_kernel(int n, double* min, double* max, double* out);
__global__ void rnorm_kernel(int n, double* mean, double* sd, double* out);
__global__ void rtnorm0_kernel(int n, double* mean, double* sd, double* l,
                        double* u, double* out); 
__global__ void rtnorm1_kernel(int n, double* mean, double* sd, double* l,
                        double* u, double* out);
__global__ void rtnorm2_kernel(int n, double* mean, double* sd, double* l,
                        double* u, double* out); 
__global__ void rtnorm3_kernel(int n, double* mean, double* sd, double* l,
                        double* u, double* out);

static __device__ inline
double rtnorm0_device(curandState_t* state, double mean, double sd,
                      double l, double u);

static __device__ inline
double rtnorm1_device(curandState_t* state, double a, double mean,
                      double sd, double l, double u); 
 
__global__ void rlba_kernel(int* n, double* b, double* A, double* mean_v,
                            double* sd_v, double* t0, double* lower,
                            double* upper, double* a, bool* c, double* RT, unsigned int* R);

void runif_entry(int *n, double *min, double *max, int *nth, double *out);
void rnorm_entry(int *n, double *mean, double *sd, int *nth, double *out);
void rtnorm_entry(int *n, double *mean, double *sd, double *l, double *u, int *nth, double *out);

void rlba_entry(int *n, double *b,double *A, double *mean_v, int *nmean_v,
                double *sd_v, int *nsd_v, double *t0, int *nth, double *RT, int *R);

__global__ void histc_kernel(double *binedge, double *rng, int *nrng, unsigned int *out);
void histc_entry(double *binedge, int ngridplus1, double *rng, int nsim,
                 int ngrid, unsigned int *out);
