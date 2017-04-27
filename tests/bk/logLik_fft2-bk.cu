
__global__ void histc_gpu(curandState_t *state, double *parameter, double *binedge,
                          int *nsim, unsigned int *histo) {
    // Each block has its own private copy of the 'cache' shared memory
    __shared__ unsigned int cache[1024];
    cache[threadIdx.x] = 0;
    __syncthreads();
    
        const int numThreads = blockDim.x * gridDim.x; // total # of threads
        const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;

        for (int i = threadID; i<nsim[0]; i += numThreads) {
          int j=0;
          double tmp=0, sim=0;

          curand_init(clock64(), i, 0, &state[i]);
          sim = parameter[0] + parameter[1]*curand_normal_double(&state[i]);
          
          while(tmp==0) {
              tmp = ((sim >= binedge[j]) && (sim < binedge[j+1])); // 0 or 1;
              j++;
          }

          // (j-1) is the bin location that this sim belongs to.
          atomicAdd( &(cache[j-1]), 1);
        }
        __syncthreads();
        // add partial histograms in each block together.
        atomicAdd( &(histo[threadIdx.x]), cache[threadIdx.x] );
}


void logLik_fft2(double *y_, double *yhat_, int *ny, int *ns,  
                 int *p, int *nThreads, double out[])
{
  arma::vec y    = getVec(y_, ny); // Convert to arma vec
  arma::vec yhat = getVec(yhat_, ns);
  double h = bwNRD0(yhat, 1.0);
  double z0   = std::min(y.min(), yhat.min()) - 3 * h;
  double z1   = std::max(y.max(), yhat.max()) + 3 * h;
  printf("[h z0 z1]: [%f %f %f]\n", h, z0, z1);
  arma::vec z = arma::linspace<arma::vec>(z0, z1, std::pow(2, *p)) ;

  /*--------------------------------------------
    Calculate the location of bin edges. (1025)
    -------------------------------------------*/
  int ngrid = 1<<*p;
  int ngridplus1 = ngrid + 1;
  int ngridminus1 = ngrid - 1;
  double dt0 = (z1-z0)/ngridminus1;
  printf("dt0: %f\n", dt0);
  arma::vec binEdges(ngridplus1);
  for(int i=0; i<ngridplus1; i++)
  {
    binEdges[i] = i < ngrid ? z0 + dt0*((double)i - 0.5) : (z0 + (double)(i-1)*dt0) +  0.5*dt0;
  }
  //arma::vec binEdges  = getEdges(z) ;

  /*--------------------------------------------
  Calculate Gaussian filter at frequency domain. (1024)
  -------------------------------------------*/
  arma::vec filter(ngrid);
  double z1minusz0 = z1 - z0;
  int halfngrid = 1<<(*p-1);
  double fil0_constant = -2*h*h*M_PI*M_PI/(z1minusz0 * z1minusz0);
  for(int i=0; i < ngrid; i++) {
           if (i < (1+halfngrid)) {
               filter[i] = std::exp(fil0_constant * i*i);
           } else { // magical flipping
               int j = 2*(i - halfngrid);
               filter[i] = filter[i-j];
           }
   }
  //arma::vec filter    = getFilter(z0, z1, h, *p) ;  // Gauss filter

  

  /*--------------------------------------------
  Calculate histogram (1024)
  -------------------------------------------*/
  curandState_t *d_state;
  unsigned int *h_histo, *d_histo;
  int *h_nsim, *d_nsim;
  double *parameter, *d_parameter, *h_binEdges, *d_binEdges;
  parameter = (double *)malloc(2 * sizeof(double));
  h_binEdges = (double *)malloc(ngridplus1 * sizeof(double));
  h_nsim = (int *)malloc(1 * sizeof(int));
  h_histo = (unsigned int *)malloc(ngrid * sizeof(unsigned int));
  parameter[0] = 0;
  parameter[1] = 1;
  for(int i=0; i<ngridplus1; i++) {h_binEdges[i] = binEdges[i];}
  h_nsim[0] = 1e5;
  for(int i=0; i<ngrid; i++) {h_histo[i] = 0;}
  cudaMalloc((void**) &d_state, h_nsim[0] * sizeof(curandState_t));
  cudaMalloc((void**) &d_parameter, 2*sizeof(double));
  cudaMalloc((void**) &d_binEdges,  ngridplus1*sizeof(double));
  cudaMalloc((void**) &d_nsim,  1*sizeof(int));
  cudaMalloc((void**) &d_histo,  ngrid*sizeof(unsigned int));
  cudaMemcpy(d_parameter, parameter, 2*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_binEdges, h_binEdges, ngridplus1*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_nsim, h_nsim, 1*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_histo, h_histo, ngrid*sizeof(unsigned int), cudaMemcpyHostToDevice);
  int nb = h_nsim[0]/(1024 + 1);
  printf("Block number: %d\n", nb);

  histc_gpu<<<nb, 1024>>>(d_state, d_parameter, d_binEdges, d_nsim, d_histo);
  cudaMemcpy(h_histo, d_histo, sizeof(unsigned int)*ngrid, cudaMemcpyDeviceToHost);

  double dt = z[1] - z[0] ;
  printf("dt: %f\n", dt);
  arma::vec signal(ngrid);
  for(int i=0; i<ngrid; i++)
  {
      signal[i] = h_histo[i] / (dt*h_nsim[0]);
  }
  
  //arma::vec signal    = density(yhat, binEdges, dt) ; // involve histc
  int nSignal         = signal.n_elem;
  int nFilter         = filter.n_elem;
  if (nSignal != nFilter) {fprintf(stderr, "signal & filter have unequal sizes\n");}

  // Define the execution configuration
  // int nthreads, nblocks;
  // nthreads = *nThreads;
  int nblocks  = (*ns)/(*nThreads) + 1;
  /* Print out how many threads are using. This is for test */
  // printf("%d\n", *nThreads);  


  // Allocate host memory for the signal and filter
  Complex *h_signal = (Complex *)malloc(sizeof(Complex) * nSignal);
  Complex *h_filter = (Complex *)malloc(sizeof(Complex) * nSignal);
  Complex *h_out    = (Complex *)malloc(sizeof(Complex) * nSignal);

  // Initialize the memory for the signal and filter
  for (unsigned int i = 0; i < nSignal; ++i)
  {
        h_signal[i].x = signal[i];
        h_signal[i].y = 0;
        h_filter[i].x = filter[i];
        h_filter[i].y = 0;
  }

  // Allocate device memory for signal and filter
  Complex *d_signal, *d_filter;

  cudaMalloc((void **)&d_signal, sizeof(Complex)*nSignal);
  cudaMalloc((void **)&d_filter, sizeof(Complex)*nSignal);

  // Copy host memory to device
  cudaMemcpy(d_signal, h_signal, sizeof(Complex)*nSignal, 
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_filter, h_filter, sizeof(Complex)*nSignal, 
             cudaMemcpyHostToDevice);

  // CUFFT plan simple API
  cufftHandle plan;
  cufftPlan1d(&plan, nSignal, CUFFT_C2C, 1);

  // Transform signal and kernel
  cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_FORWARD);

  // Multiply the coefficients together and normalize the result
  ComplexPointwiseMulAndScale<<<nblocks,*nThreads>>>(d_signal, d_filter, nSignal, 1.0f / nSignal);

  // Transform signal back
  // printf("Transforming signal back cufftExecC2C\n");
  cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_INVERSE);

  // Copy device memory to host
  cudaMemcpy(h_out, d_signal, sizeof(Complex)*nSignal, cudaMemcpyDeviceToHost);

  arma::vec PDF_smoothed(nSignal) ;
  for(int i=0; i<nSignal; i++)
  {
    PDF_smoothed[i] = h_out[i].x;  // Extract real part of the complex numbers 
  }

  arma::vec PDF;   // Interpolate the grid likelihood to the data
  arma::interp1(z, PDF_smoothed, y, PDF);
  arma::vec PDF_tmp = pmax(PDF, std::pow(10, -10)) ;
  // *out = arma::accu(arma::log(PDF_tmp));
  // *out = arma::log(PDF_tmp);
  for(arma::vec::iterator i=PDF_tmp.begin(); i!=PDF_tmp.end(); ++i)
  {
    int idx  = std::distance(PDF_tmp.begin(), i);
    out[idx] = std::log(*i);
  }

  // cleanup memory
  free(h_signal);
  free(h_filter);
  free(h_out);
  cudaFree(d_signal);
  cudaFree(d_filter);
}

void logLik_fft2_new(double *y_, double *yhat_, int *ny, int *ns,  
                 int *p, int *nThread, double out[])
{
    double *d_rngnums, *d_rngarg, *h_rngarg, *pinned_rngnums;
    cudaHostAlloc((void**)&pinned_rngnums, *ns * sizeof(double), cudaHostAllocDefault);
    h_rngarg   = (double *)malloc(3 * sizeof(double));
    h_rngarg[0] = *ns;
    h_rngarg[1] = 0;
    h_rngarg[2] = 1;
    int nBlk    = (*ns)/(*nThread) + 1;
    printf("Blocks, threads %d %d\n", nBlk, *nThread);

    cudaEvent_t start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    float elapsedTime;

    cudaMalloc((void**) &d_rngnums, *ns * sizeof(double));
    cudaMalloc((void**) &d_rngarg,     3 * sizeof(double));
    cudaMemcpy(d_rngarg, h_rngarg,     3 * sizeof(double), cudaMemcpyHostToDevice);

    cudaEventRecord( start, 0);
    rnorm2<<<nBlk, *nThread>>>(d_rngarg, d_rngnums);
    cudaEventRecord( stop, 0);
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &elapsedTime, start, stop );
    printf("Time for rnorm2: %f ms\n", elapsedTime);

    cudaEventRecord( start, 0);
    cudaMemcpy(pinned_rngnums, d_rngnums, *ns * sizeof(double), cudaMemcpyDeviceToHost);
    cudaEventRecord( stop, 0);
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &elapsedTime, start, stop );
    printf("Time for pinned memory out: %f ms\n", elapsedTime);

    double sec;
    clock_t b = clock();
    arma::vec yhat = getVec(pinned_rngnums, ns);
    cudaFreeHost(pinned_rngnums);
    cudaFree(d_rngnums);
    cudaFree(d_rngarg);
    free(h_rngarg);

    sec = (double)(clock() - b) / (double) CLOCKS_PER_SEC;
    printf("Time for getting yhat: %f ms\n", 1000*sec);

  arma::vec y    = getVec(y_, ny); // Convert to arma vec
  double h = bwNRD0(yhat, 1.0);
  double z0   = std::min(y.min(), yhat.min()) - 3 * h;
  double z1   = std::max(y.max(), yhat.max()) + 3 * h;
  printf("[h z0 z1]: [%f %f %f]\n", h, z0, z1);
  arma::vec z = arma::linspace<arma::vec>(z0, z1, std::pow(2, *p)) ;

  /*--------------------------------------------
    Calculate the location of bin edges. (1025)
    -------------------------------------------*/
  int ngrid = 1<<*p;
  int ngridplus1 = ngrid + 1;
  int ngridminus1 = ngrid - 1;
  double dt0 = (z1-z0)/ngridminus1;
  arma::vec binEdges(ngridplus1);
  for(int i=0; i<ngridplus1; i++)
  {
    binEdges[i] = i < ngrid ? z0 + dt0*((double)i - 0.5) : (z0 + (double)(i-1)*dt0) +  0.5*dt0;
  }
  //arma::vec binEdges  = getEdges(z) ;

  /*--------------------------------------------
  Calculate Gaussian filter at frequency domain. (1024)
  -------------------------------------------*/
  arma::vec filter(ngrid);
  double z1minusz0 = z1 - z0;
  int halfngrid = 1<<(*p-1);
  double fil0_constant = -2*h*h*M_PI*M_PI/(z1minusz0 * z1minusz0);
  for(int i=0; i < ngrid; i++) {
           if (i < (1+halfngrid)) {
               filter[i] = std::exp(fil0_constant * i*i);
           } else { // magical flipping
               int j = 2*(i - halfngrid);
               filter[i] = filter[i-j];
           }
   }
  //arma::vec filter    = getFilter(z0, z1, h, *p) ;  // Gauss filter
  

  /*--------------------------------------------
  Calculate histogram (1024)
  -------------------------------------------*/
  //curandState_t *d_state;
  //unsigned int *h_histo;
  unsigned int *d_histo;
  unsigned int *pinned_histo;
  int *h_nsim, *d_nsim;
  double *parameter, *d_parameter, *h_binEdges, *d_binEdges;
  parameter = (double *)malloc(2 * sizeof(double));
  h_binEdges = (double *)malloc(ngridplus1 * sizeof(double));
  h_nsim = (int *)malloc(1 * sizeof(int));

  cudaHostAlloc((void**)&pinned_histo, ngrid * sizeof(unsigned int), cudaHostAllocDefault);
  //h_histo = (unsigned int *)malloc(ngrid * sizeof(unsigned int));

  parameter[0] = 0;
  parameter[1] = 1;
  for(int i=0; i<ngridplus1; i++) {h_binEdges[i] = binEdges[i];}
  h_nsim[0] = 1e5;

  //for(int i=0; i<ngrid; i++) {h_histo[i] = 0;}
  for(int i=0; i<ngrid; i++) {pinned_histo[i] = 0;}

  //cudaMalloc((void**) &d_state, h_nsim[0] * sizeof(curandState_t));
  cudaMalloc((void**) &d_parameter, 2*sizeof(double));
  cudaMalloc((void**) &d_binEdges,  ngridplus1*sizeof(double));
  cudaMalloc((void**) &d_nsim,  1*sizeof(int));
  cudaMalloc((void**) &d_histo,  ngrid*sizeof(unsigned int));
  cudaMemcpy(d_parameter, parameter, 2*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_binEdges, h_binEdges, ngridplus1*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_nsim, h_nsim, 1*sizeof(int), cudaMemcpyHostToDevice);

  //cudaMemcpy(d_histo, h_histo, ngrid*sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_histo, pinned_histo, ngrid*sizeof(unsigned int), cudaMemcpyHostToDevice);

  int nb = h_nsim[0]/(1024 + 1);
  printf("Block number: %d\n", nb);
  

    cudaEventRecord( start, 0);
    histc_gpu<<<nb, 1024>>>(d_parameter, d_binEdges, d_nsim, d_histo);
    cudaEventRecord( stop, 0);
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &elapsedTime, start, stop );
    printf("Time for histc_gpu: %f ms\n", elapsedTime);

    cudaEventRecord( start, 0);
    cudaMemcpy(pinned_histo, d_histo, sizeof(unsigned int)*ngrid, cudaMemcpyDeviceToHost);
    cudaEventRecord( stop, 0);
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &elapsedTime, start, stop );
    printf("Time for getting unpinned h_histo: %f ms\n", elapsedTime);


  double dt = z[1] - z[0] ;
  arma::vec signal(ngrid);
  for(int i=0; i<ngrid; i++)
  {
      //signal[i] = h_histo[i] / (dt*h_nsim[0]);
      signal[i] = pinned_histo[i] / (dt*h_nsim[0]);
  }

  cudaFreeHost(pinned_histo);

  int nSignal         = signal.n_elem;
  int nFilter         = filter.n_elem;
  if (nSignal != nFilter) {fprintf(stderr, "signal & filter have unequal sizes\n");}

  // Define the execution configuration
  // int nthreads, nblocks;
  // nthreads = *nThreads;
  int nblocks  = (*ns)/(*nThread) + 1;
  /* Print out how many threads are using. This is for test */
  // printf("%d\n", *nThreads);  


  // Allocate host memory for the signal and filter
  Complex *h_signal = (Complex *)malloc(sizeof(Complex) * nSignal);
  Complex *h_filter = (Complex *)malloc(sizeof(Complex) * nSignal);
  Complex *h_out    = (Complex *)malloc(sizeof(Complex) * nSignal);

  // Initialize the memory for the signal and filter
  for (unsigned int i = 0; i < nSignal; ++i)
  {
        h_signal[i].x = signal[i];
        h_signal[i].y = 0;
        h_filter[i].x = filter[i];
        h_filter[i].y = 0;
  }

  // Allocate device memory for signal and filter
  Complex *d_signal, *d_filter;

  cudaMalloc((void **)&d_signal, sizeof(Complex)*nSignal);
  cudaMalloc((void **)&d_filter, sizeof(Complex)*nSignal);

  // Copy host memory to device
  cudaMemcpy(d_signal, h_signal, sizeof(Complex)*nSignal, 
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_filter, h_filter, sizeof(Complex)*nSignal, 
             cudaMemcpyHostToDevice);

  // CUFFT plan simple API
  cufftHandle plan;
  cufftPlan1d(&plan, nSignal, CUFFT_C2C, 1);

  // Transform signal and kernel
  cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_FORWARD);

  // Multiply the coefficients together and normalize the result
  ComplexPointwiseMulAndScale<<<nblocks,*nThread>>>(d_signal, d_filter, nSignal, 1.0f / nSignal);

  // Transform signal back
  // printf("Transforming signal back cufftExecC2C\n");
  cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_INVERSE);

  // Copy device memory to host
  cudaMemcpy(h_out, d_signal, sizeof(Complex)*nSignal, cudaMemcpyDeviceToHost);

  arma::vec PDF_smoothed(nSignal) ;
  for(int i=0; i<nSignal; i++)
  {
    PDF_smoothed[i] = h_out[i].x;  // Extract real part of the complex numbers 
  }

  arma::vec PDF;   // Interpolate the grid likelihood to the data
  arma::interp1(z, PDF_smoothed, y, PDF);
  arma::vec PDF_tmp = pmax(PDF, std::pow(10, -10)) ;
  // *out = arma::accu(arma::log(PDF_tmp));
  // *out = arma::log(PDF_tmp);
  for(arma::vec::iterator i=PDF_tmp.begin(); i!=PDF_tmp.end(); ++i)
  {
    int idx  = std::distance(PDF_tmp.begin(), i);
    out[idx] = std::log(*i);
  }

  // cleanup memory
  free(h_signal);
  free(h_filter);
  free(h_out);
  cudaFree(d_signal);
  cudaFree(d_filter);
}

  arma::cx_vec signal_fft = arma::fft(signal);
  arma::cx_vec temp = filter % signal_fft;
  arma::cx_vec temp2 = arma::ifft(temp);
  arma::vec PDF0 = arma::real(temp2);
   
  arma::vec PDF;
  arma::interp1(z, PDF0, y, PDF);
  arma::vec PDF_ = pmax(PDF, std::pow(10, -5)) ;

  for(int k=0; k<*ny; k++) { out[k] = PDF_[k]; }


__global__ void pda_rnorm(curandState_t *state, double *parameter, double *binedge,
                          int *nsim, unsigned int *histo, 
                          double *out) {
    // Each block has its own private copy of the 'cache' shared memory
    __shared__ unsigned int cache[1024];
    cache[threadIdx.x] = 0;
    __syncthreads();
    
        const int numThreads = blockDim.x * gridDim.x; // total # of threads
        const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;

        for (int i = threadID; i<nsim[0]; i += numThreads) {
          int j=0;
          double tmp=0, sim=0;

          curand_init(clock64(), i, 0, &state[i]);
          sim = parameter[0] + parameter[1]*curand_normal_double(&state[i]);
          
          while(tmp==0) {
              tmp = ((sim >= binedge[j]) && (sim < binedge[j+1])); // 0 or 1;
              j++;
          }

          // (j-1) is the bin location that this sim belongs to.
          atomicAdd( &(cache[j-1]), 1);
        }
        __syncthreads();
        // add partial histograms in each block together.
        atomicAdd( &(histo[threadIdx.x]), cache[threadIdx.x] );
}

void pda_rnorm_gpu(double *y, int *ny, int *nsim, int *nThreads, int *nBlk, double *out)
{
   arma::vec data = getVec(y, ny); // Convert to arma vec
   double h = bwNRD0(data, 0.8);
   int p = 10;

   curandState_t *d_state;
   int *ngrid, *d_nsim;
   unsigned int *h_histo, *d_histo;
   double *h_z0, *h_z1, *binedge, *parameter, *h_unit, *filter;
   double *d_binedge, *d_parameter, *d_out;

   ngrid = (int *)malloc((int)(1) * sizeof(int));
   *ngrid = 1<<p;
   int ngridminus1 = *ngrid - 1;
   int ngridplus1  = *ngrid + 1;
   int halfngrid   = 0.5*(*ngrid);

   h_z0      = (double *)malloc(1.0 * sizeof(double));
   h_z1      = (double *)malloc(1.0 * sizeof(double));
   filter    = (double *)malloc(*ngrid * sizeof(double));
   binedge   = (double *)malloc(ngridplus1 * sizeof(double));
   parameter = (double *)malloc(2.0 * sizeof(double));
   h_unit    = (double *)malloc(1.0 * sizeof(double));
   h_histo   = (unsigned int *)malloc(*ngrid * sizeof(unsigned int));
   for(int i=0; i<*ngrid; i++) { h_histo[i] = 0;}

   *h_z0 = arma::min(data) - 3*h;
   *h_z1 = arma::max(data) + 3*h;
   double z1minusz0 = *h_z1 - *h_z0;
   double dt = z1minusz0 / (double)ngridminus1;
   printf("z0 and z1, [%f %f]\n", *h_z0, *h_z1);
   

   parameter[0] = 0;
   parameter[1] = 1;

   // Calculate the location of bin edges. (1025)
   for(int i=0; i<ngridplus1; i++) {
       binedge[i] = i < *ngrid ? *h_z0 + dt*((double)i - 0.5) :
           (*h_z0 + (double)(i-1)*dt) +  0.5*dt;
   }

   // Gaussian filter at frequency domain. (1024)
   double fil0_constant = -2*h*h*M_PI*M_PI/(z1minusz0 * z1minusz0);
   for(int i=0; i<*ngrid; i++) {
           if (i < (1+halfngrid)) {
               filter[i] = std::exp(fil0_constant * i*i);
           } else { // magical flipping
               int j = 2*(i - halfngrid);
               filter[i] = filter[i-j];
           }
   }

   cudaMalloc((void**) &d_state, *nsim * sizeof(curandState_t));
   cudaMalloc((void**) &d_out,   *ngrid * sizeof(double));

   cudaMalloc((void**) &d_binedge, (ngridplus1) * sizeof(double));
   cudaMalloc((void**) &d_parameter, (2) * sizeof(double));
   cudaMalloc((void**) &d_nsim, (1) * sizeof(int));
   cudaMalloc((void**) &d_histo, *ngrid * sizeof(unsigned int));

   cudaMemcpy(d_binedge, binedge, sizeof(double)*(ngridplus1), cudaMemcpyHostToDevice);
   cudaMemcpy(d_parameter, parameter, sizeof(double)*(2), cudaMemcpyHostToDevice);
   cudaMemcpy(d_histo, h_histo, sizeof(unsigned int)*(*ngrid), cudaMemcpyHostToDevice);

   cudaMemcpy(d_nsim, nsim, sizeof(int)*(1), cudaMemcpyHostToDevice);

   pda_rnorm<<<*nBlk, *nThreads>>>(d_state, d_parameter, d_binedge, d_nsim,
                                  d_histo, d_out);
   cudaMemcpy(h_histo, d_histo, sizeof(unsigned int)*(*ngrid), cudaMemcpyDeviceToHost);

   *h_unit = dt * (*nsim);
   printf("%f\n", *h_unit);

   double pdf[*ngrid];
   for(int k=0; k<(*ngrid); k++) {
       pdf[k] = h_histo[k] / (*h_unit);
       //printf("histo[%d]: %d\n", k, h_histo[k]);
   }

   arma::vec signal = getVec(pdf, ngrid); // Convert to arma vec
   arma::cx_vec signal_fft = arma::fft(signal);
   arma::vec filter_vec = getVec(filter, ngrid);
   arma::cx_vec temp = filter_vec % signal_fft;
   arma::cx_vec temp2 = arma::ifft(temp);
   arma::vec PDF0 = arma::real(temp2);
   
   arma::vec PDF;
   arma::vec z = arma::linspace<arma::vec>(*h_z0, *h_z1, 1<<(int)p);
   arma::interp1(z, PDF0, data, PDF);
   arma::vec PDF_ = pmax(PDF, std::pow(10, -5)) ;

   for(int k=0; k<*ny; k++) {
       out[k] = PDF_[k];
   }

   free(filter);
   free(binedge);
   free(parameter);
   free(h_histo);
   cudaFree(d_state);
   cudaFree(d_binedge);
   cudaFree(d_parameter);
   cudaFree(d_nsim);
   cudaFree(d_histo);
   cudaFree(d_out);
}



