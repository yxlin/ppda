// a CUDA kernel function to compute a histogram of the given random number 
// (double rng) and defined bin edges (double binedge).   
__global__ void histc_kernel(double *binedge, double *rng, unsigned int *nrng, 
  unsigned int *out);

// a CUDA kernel function to compute a histogram. This function takes float 
// binedge and float rng    
__global__ void histc_kernel(float *binedge, float *rng, unsigned int *nrng,
  unsigned int *out);

// A C function to access histc_kernel  
void histc_entry(double *binedge, double *rng, int nrng, int ngrid, 
  unsigned int *out);

// node 1 probability density function of LBA model. node 1 refers to the 
// density for the first accumulator. 
void n1PDF(double *RT0, int *ndata, int *n, double *b, double *A, 
  double *mean_v, int *nmean_v, double *sd_v, int *nsd_v, double *t0, 
  int *nth, bool *debug, double *out);

