void rtnorm_alg0(int *n, double *mean, double *sd, double *lower, double *upper, 
   int *nThreads, double *out)
{
  curandState_t *d_state;  
  double *d_out, *d_mean, *d_sd, *d_l, *d_u; // lower and upper bounds
  int nthreads = *nThreads;   // Define the execution configuration
  int nblocks  = (*n)/(nthreads) + 1;

  // Initialise the memory 
  double *h_mean = (double *)malloc(sizeof(double) * (1));
  double *h_sd   = (double *)malloc(sizeof(double) * (1));
  double *h_u    = (double *)malloc(sizeof(double) * (1));
  double *h_l    = (double *)malloc(sizeof(double) * (1));
  h_mean[0]      = *mean;
  h_sd[0]        = *sd;
  h_l[0]         = *lower;
  h_u[0]         = *upper;
  
  /* allocate space on the GPU for the random states and gpu rng */
  cudaMalloc((void**) &d_state, (*n) * sizeof(curandState_t));
  cudaMalloc((void**) &d_out,   (*n) * sizeof(double));
  cudaMalloc((void**) &d_mean,  (1)  * sizeof(double));
  cudaMalloc((void**) &d_sd,    (1)  * sizeof(double));
  cudaMalloc((void**) &d_l,     (1)  * sizeof(double));
  cudaMalloc((void**) &d_u,     (1)  * sizeof(double));
  
  cudaMemcpy(d_mean, h_mean, sizeof(double)*(1), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sd,   h_sd,   sizeof(double)*(1), cudaMemcpyHostToDevice);
  cudaMemcpy(d_l,    h_l,    sizeof(double)*(1), cudaMemcpyHostToDevice);
  cudaMemcpy(d_u,    h_u,    sizeof(double)*(1), cudaMemcpyHostToDevice);

  // Invalid situation; return NAN
  double stdlower = (h_l[0] - *mean) / *sd;
  double stdupper = (h_u[0] - *mean) / *sd;
  printf("stdlower %f, stdupper %f\n", stdlower, stdupper);
  bool a0 = stdlower > stdupper;

  // Algorithm (1): Use Naive A-R method
  bool a1 = (stdlower < 0 && stdupper==INFINITY)  || 
      (stdlower == -INFINITY && stdupper > 0) ||
      (stdlower == INFINITY  && stdupper == INFINITY && stdlower < 0 && stdupper > 0 &&
      (stdupper - stdlower) > SQRT_2PI);
  
  // Algorithm (2): Use Proposition 2.3 with only lower truncation. upper==INFINITY
  double term1_a2 = stdlower ;
  double term2_a2 = 2 * std::sqrt(M_E) / (stdlower + std::sqrt(stdlower * stdlower + 4.0));
  double term3_a2 = std::exp( 0.25 * (2 * stdlower - stdlower * 
     std::sqrt(stdlower * stdlower + 4.0)));
  double eq_a2 = term1_a2 + term2_a2 * term3_a2;
  bool a2 = stdlower >= 0 && stdupper > eq_a2;

  // Algorithm (3): Use -x ~ N_+ (-mu, -mu^+, sigma^2) on page 123. lower==-INFINITY
  double term1_a3 = -stdupper ;
  double term2_a3 = 2 * std::sqrt(M_E) / (-stdupper + std::sqrt(stdupper * stdupper + 4.0));
  double term3_a3 = std::exp( 0.25 * (2 * stdupper - stdupper * 
     std::sqrt(stdupper * stdupper + 4.0)));
  double eq_a3 = term1_a3 + term2_a3 * term3_a3;
  bool a3 = (stdupper <= 0) && ( -stdlower > eq_a3);

  /* GPU initializes all of the random states and generates random #s*/
  if(a0) {
       printf("upper bound must be greater than lower bound!");
  } else if (a1) {
       printf("Algorithm 1\n");
       rtnorm_alg1<<<nblocks, nthreads>>>(time(0), d_state, *n, d_l, d_u, d_out);
  } else if (a2) {
       printf("Algorithm 2\n");
       rtnorm_alg2<<<nblocks, nthreads>>>(time(0), d_state, *n, d_l, d_u, d_out);
  } else if (a3) {
       printf("Algorithm 3\n");
       rtnorm_alg3<<<nblocks, nthreads>>>(time(0), d_state, *n, d_l, d_u, d_out);
  } else {
       printf("Algorithm 4\n");
       rtnorm_alg4<<<nblocks, nthreads>>>(time(0), d_state, *n, d_l, d_u, d_out);
  }
  
  /* copy the random numbers back */
  cudaMemcpy(out, d_out, (*n) * sizeof(double), cudaMemcpyDeviceToHost);
  free(h_mean);
  free(h_sd);
  free(h_l);
  free(h_u);
  cudaFree(d_state);
  cudaFree(d_out);
  cudaFree(d_mean);
  cudaFree(d_sd);
  cudaFree(d_l);
  cudaFree(d_u);
}

/*
void dlba_gpu(double *RT0, double *RT1, int *nRT0, int *nRT1,
              int *nsim, double *b, double *A,
              double *v, int *nv, double *sv, double *t0,
              int *nThread, double *den0, double *den1) {
        bool debug=false;

        double *h_RT;
        int *h_R;
        cudaHostAlloc((void**)&h_RT, *nsim * sizeof(double), cudaHostAllocDefault);
        cudaHostAlloc((void**)&h_R,  *nsim * sizeof(int),    cudaHostAllocDefault);
        rlba_gpu(nsim, b, A, v, nv, sv, t0, nThread, h_RT, h_R);

        /////////////////////////////////
        int *h_nc;
        h_nc = (int *)malloc(nv[0] * sizeof(int));
        for(int i=0; i<nv[0]; i++) { h_nc[i] = 0; }

        for(int i=0; i<*nsim; i++) {
                if(h_R[i] == 1) {
                    h_nc[0]++;
                } else {
                    h_nc[1]++;
                }
        }

        /////////////////////////////////
        int j=0, k=0;
        arma::vec sRT0(h_nc[0]);
        arma::vec sRT1(h_nc[1]);
        double *h_RT0, *h_RT1;
        cudaHostAlloc((void**)&h_RT0, h_nc[0] * sizeof(double), cudaHostAllocDefault);
        cudaHostAlloc((void**)&h_RT1, h_nc[1] * sizeof(double), cudaHostAllocDefault);
        for(int i=0; i<*nsim; i++) {
                if(h_R[i] == 1) {
                    sRT0[j] = h_RT[i];
                    h_RT0[j] = h_RT[i];
                    j++;
                } else {
                    sRT1[k] = h_RT[i];
                    h_RT1[k] = h_RT[i];
                    k++;
                }
        }

        cudaFreeHost(h_RT);
        cudaFreeHost(h_R);

        /////////////////////////////////
        double *h_sd, *h_h, *h_z0, *h_z1;
        h_sd = (double *)malloc(nv[0] * sizeof(double));
        h_h  = (double *)malloc(nv[0] * sizeof(double));
        h_z0 = (double *)malloc(nv[0] * sizeof(double));
        h_z1 = (double *)malloc(nv[0] * sizeof(double));

        h_sd[0] = arma::stddev(sRT0) ;
        h_sd[1] = arma::stddev(sRT1) ;
        h_h[0]  = 0.8*h_sd[0]*std::pow((double)h_nc[0], -0.2);
        h_h[1]  = 0.8*h_sd[1]*std::pow((double)h_nc[1], -0.2);

        h_z0[0]  = sRT0.min() - 3*h_h[0];
        h_z1[0]  = sRT0.max() + 3*h_h[0];

        h_z0[1]  = sRT1.min() - 3*h_h[1];
        h_z1[1]  = sRT1.max() + 3*h_h[1];

        if(debug) {
          printf("choice 0 and 1 [%f %f]\n", h_z0[0], h_z0[1]);
          printf("choice 0 and 1 [%f %f]\n", h_z1[0], h_z1[1]);
        }

        /////////////////////////////////
        //  Calculate bin edges. (1025) 
        double *h_dt, *h_binedge0, *h_binedge1;
        h_dt = (double *)malloc(nv[0] * sizeof(double));
        int ngrid = 1024; // always use 1024
        int ngridplus1  = ngrid + 1;
        int ngridminus1 = ngrid - 1;
        h_dt[0] = (h_z1[0] - h_z0[0])/ngridminus1;
        h_dt[1] = (h_z1[1] - h_z0[1])/ngridminus1;
        cudaHostAlloc((void**)&h_binedge0, ngridplus1 * sizeof(double), cudaHostAllocDefault);
        cudaHostAlloc((void**)&h_binedge1, ngridplus1 * sizeof(double), cudaHostAllocDefault);
        for(int i=0; i<ngridplus1; i++) {
           h_binedge0[i] = i < ngrid ? h_z0[0] + h_dt[0]*((double)i - 0.5) :
                   (h_z0[0] + (double)(i-1)*h_dt[0]) +  0.5*h_dt[0];
           h_binedge1[i] = i < ngrid ? h_z0[1] + h_dt[1]*((double)i - 0.5) :
                   (h_z0[1] + (double)(i-1)*h_dt[1]) +  0.5*h_dt[1];
        }

        if(debug) { 
            for(int i=0; i<10; i++) { printf("binedge0[%d]: %f\n", i, h_binedge0[i]); }
            printf("\n");
            for(int i=0; i<10; i++) { printf("binedge1[%d]: %f\n", i, h_binedge1[i]); }
        }

        ///////////////////////////////// Gaussian filter at frequency domain. (1024)
        arma::vec filter0(ngrid);
        arma::vec filter1(ngrid);
        double *h_z1minusz0;
        int halfngrid = 512;
        h_z1minusz0 = (double *)malloc(nv[0] * sizeof(double));
        h_z1minusz0[0] = h_z1[0] - h_z0[0];
        h_z1minusz0[1] = h_z1[1] - h_z0[1];
        double fil0_constant0 = -2*h_h[0]*h_h[0]*M_PI*M_PI / (h_z1minusz0[0]*h_z1minusz0[0]);
        double fil0_constant1 = -2*h_h[1]*h_h[1]*M_PI*M_PI / (h_z1minusz0[1]*h_z1minusz0[1]);
        for(int i=0; i<ngrid; i++) {
           if (i < (1+halfngrid)) {
               filter0[i] = std::exp(fil0_constant0 * i*i);
               filter1[i] = std::exp(fil0_constant1 * i*i);
           } else { 
               int j = 2*(i - halfngrid); // magical flipping
               filter0[i] = filter0[i-j];
               filter1[i] = filter1[i-j];
           }
        }

        if(debug) {
          for(int i=0; i<10; i++) { printf("filter0[%d]: %f\n", i, filter0[i]); }
          printf("\n");
          for(int i=500; i<512; i++) { printf("filter0[%d]: %f\n", i, filter0[i]); }
          printf("\n");
          for(int i=0; i<10; i++) { printf("filter1[%d]: %f\n", i, filter1[i]); }
          printf("\n");
          for(int i=500; i<512; i++) { printf("filter1[%d]: %f\n", i, filter1[i]); }
        }

        ////////////Get histogram (1024) 
        unsigned int *h_histo0, *h_histo1;
        cudaHostAlloc((void**)&h_histo0, ngrid * sizeof(unsigned int), cudaHostAllocDefault);
        cudaHostAlloc((void**)&h_histo1, ngrid * sizeof(unsigned int), cudaHostAllocDefault);
        for(int i=0; i<ngrid; i++) {h_histo0[i] = 0; h_histo1[i]=0;}
        //void histc(double *binedge, int ngridplus1, double *rng, int nsim, unsigned int *histo, int ngrid) {
        histc(h_binedge0, ngridplus1, h_RT0, h_nc[0], h_histo0, ngrid);
        histc(h_binedge1, ngridplus1, h_RT1, h_nc[1], h_histo1, ngrid);

        
        // if(debug) {
        //   printf("histo0 1 to 1024: \n");
        //   for(int i=0; i<ngrid; i++) { printf("%d\t", h_histo0[i]); }
        //   printf("\n");
        // }
        // 
        // if(debug) {
        //   printf("histo1 1 to 1024: \n");
        //   for(int i=0; i<ngrid; i++) { printf("%d\t", h_histo1[i]); }
        //   printf("\n");
        // }
        

       ///////////// FFT 
       arma::vec signal0(ngrid);
       arma::vec signal1(ngrid);
       for(int i=0; i<ngrid; i++)
       {
           //signal0[i] = h_histo0[i] / (h_dt[0]*h_nc[0]);
           //signal1[i] = h_histo1[i] / (h_dt[1]*h_nc[1]);
           signal0[i] = h_histo0[i] / (h_dt[0]*(*nsim));
           signal1[i] = h_histo1[i] / (h_dt[1]*(*nsim));
       }
       arma::vec PDF0 = arma::real(arma::ifft(filter0 % arma::fft(signal0))) ; // smoothed
       arma::vec PDF1 = arma::real(arma::ifft(filter1 % arma::fft(signal1))) ; // smoothed

       arma::vec z_choice0 = arma::linspace<arma::vec>(h_z0[0], h_z1[0], 1024);
       arma::vec z_choice1 = arma::linspace<arma::vec>(h_z0[1], h_z1[1], 1024);

       arma::vec y0(*nRT0);
       arma::vec y1(*nRT1);
       for(int i=0; i<*nRT0; i++) { y0[i]=RT0[i];}
       for(int i=0; i<*nRT1; i++) { y1[i]=RT1[i];}

       arma::vec PDF_out0, PDF_out1;
       arma::interp1(z_choice0, PDF0, y0, PDF_out0);
       arma::interp1(z_choice1, PDF1, y1, PDF_out1);
       //PDF_ = pmax(PDF, std::pow(10, -5)) ;

       for(int i=0; i<*nRT0; i++) {
           if(PDF_out0[i] < 1e-5) {PDF_out0[i] = 1e-5;}
       }

       for(int i=0; i<*nRT1; i++) {
           if(PDF_out1[i] < 1e-5) {PDF_out1[i] = 1e-5;}
       }

       //////////// Finish off 
       for(int i=0; i<*nRT0; i++) {
           den0[i] = PDF_out0[i];
       }

       for(int i=0; i<*nRT1; i++) {
           den1[i] = PDF_out1[i];
       }
 }

*/


/* this GPU kernel function is used to initialize the random states */
/*
__global__ void init(unsigned int seed, curandState_t* states, int n) {
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < n; i += numThreads)
    {
          curand_init(seed, i, 0, &states[i]);
    }
}
*/

        /*
        if(c0[0]) {
            rate0 = rtnorm0_device(&state, mean_v[0], sd_v[0], lower[0], upper[0]);
        } else if (c0[1]) {
            rate0 = rtnorm1_device(&state, a0[0], mean_v[0], sd_v[0], lower[0], upper[0]);
        } else if (c0[2]) {
            rate0 = rtnorm2_device(&state, a0[1], mean_v[0], sd_v[0], lower[0], upper[0]);
        } else {
            rate0 = rtnorm3_device(&state, mean_v[0], sd_v[0], lower[0], upper[0]);
        }

        if(c1[0]) {
            rate1 = rtnorm0_device(&state, mean_v[1], sd_v[1], lower[1], upper[1]);
        } else if (c1[1]) {
            rate1 = rtnorm1_device(&state, a1[0], mean_v[1], sd_v[1], lower[0], upper[1]);
        } else if (c1[2]) {
            rate1 = rtnorm2_device(&state, a1[1], mean_v[1], sd_v[1], lower[0], upper[1]);
        } else {
            rate1 = rtnorm3_device(&state, mean_v[1], sd_v[1], lower[0], upper[1]);
        }
        */



