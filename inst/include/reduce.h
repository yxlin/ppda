unsigned int nextPow2(unsigned int x);
__global__ void min_kernel(double *g_idata, double *g_odata);
__global__ void min_kernel(float *g_idata, float *g_odata);

__global__ void max_kernel(double *g_idata, double *g_odata);
__global__ void max_kernel(float *g_idata, float *g_odata);

__global__ void minmax_kernel(double *g_idata, double *g_odata); 
__global__ void minmax_kernel(float *g_idata, float *g_odata); 

__global__ void sum_kernel(double *g_idata, double *g_odata);
__global__ void sum_kernel(float *g_idata, float *g_odata);

__global__ void squareSum_kernel(unsigned int* n, double *g_idata, double *g_odata); 
__global__ void squareSum_kernel(unsigned int* n, float *g_idata, float *g_odata); 

__global__ void n1min_kernel(float *RT0, float *out);
__global__ void n1max_kernel(float *RT0, float *out);
  
__global__ void count_kernel(unsigned int *n, unsigned int *R, unsigned int *out);

void sum_entry(double *x, int *nx, bool *debug, double *out);
void sqsum_entry(double *x, int *nx, bool *debug, double *out);
void sd_entry(double *x, int *nx, bool *debug, double *out);
void min_entry(double *x, int *nx, bool *debug, double *out);
void max_entry(double *x, int *nx, bool *debug, double *out);
void minmax_entry(double *x, int *nx, bool *debug, double *out);
void count_entry(unsigned int *n, int *R, int *nth, bool *debug, double *out); 
void n1min_entry(double *RT0, int *nx, bool *debug, double *out);
