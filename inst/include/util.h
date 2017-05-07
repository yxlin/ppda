#include <armadillo> // Armadillo library

#define SQRT_2PI   2.5066282746310007e+0 /* sqrt(2 x pi) */
#define M_E	   2.7182818284590452354	/* e */
#define MAX_BLOCK_DIM_SIZE 65535
#define CUDART_INF_F            __int_as_float(0x7f800000)
#define CUDART_INF __longlong_as_double(0x7ff0000000000000ULL) /* work insider device */

unsigned int nextPow2(unsigned int x);
void summary(int *nsim, unsigned int *d_R, float *d_RT, float *out);
void histc(int *nsim, int ngrid, float *h_binedge, float *d_RT, unsigned int *h_hist);
