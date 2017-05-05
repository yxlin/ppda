__global__ void histc_kernel(double *binedge, double *rng, int *nrng, unsigned int *out);
void histc_entry(double *binedge, double *rng, int nrng, int ngrid, unsigned int *out);
void n1PDF(double *RT0, int *ndata, int *n, double *b, double *A, double *mean_v, int *nmean_v,
  double *sd_v, int *nsd_v, double *t0, int *nth, bool *debug, double *out);
void n1PDF_test(double *RT0, int *ndata, int *n, double *b, double *A, double *mean_v, int *nmean_v,
  double *sd_v, int *nsd_v, double *t0, int *nth, bool *debug, double *out);
