#include <armadillo> // Armadillo library

unsigned int nextPow2(unsigned int x);
void summary(int *nsim, unsigned int *d_R, float *d_RT, float *out);
void histc(int *nsim, int ngrid, float *h_binedge, float *d_RT, unsigned int *h_hist);
