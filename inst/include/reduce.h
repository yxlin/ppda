unsigned int nextPow2(unsigned int x);
__global__ void min_kernel(double *g_idata, double *g_odata);
__global__ void max_kernel(double *g_idata, double *g_odata);
__global__ void count_kernel(unsigned int *n, unsigned int *R, unsigned int *out);
__global__ void minmax_kernel(double *g_idata, double *g_odata); 
__global__ void sum_kernel(double *g_idata, double *g_odata);
__global__ void squareSum_kernel(unsigned int* n, double *g_idata, double *g_odata); 

void count_entry(unsigned int *n, int *R, int *nth, bool *debug, double *out); 