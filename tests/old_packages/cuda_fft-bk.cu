extern "C" void logLik_fft(double *y_, double *yhat_, int *ny, 
  int *ns, double *h_, double *m_, int *p, int *nThreads, 
  double *out);

// Forward declear CUDA's C functions
typedef float2 Complex; // Extract cuda's complex data type
static __device__ __host__ inline Complex ComplexScale(Complex, float);
static __device__ __host__ inline Complex ComplexMul(Complex, Complex);
static __global__ void ComplexPointwiseMulAndScale(Complex *, 
  const Complex *, int, float);


/* -------------------------------------------------------------------------  
 Complex operations 
------------------------------------------------------------------------- */
// Complex scale
static __device__ __host__ inline Complex ComplexScale(Complex a, float s)
{
  Complex c;
  c.x = s * a.x;
  c.y = s * a.y;
  return c;
}

// Complex multiplication
static __device__ __host__ inline Complex ComplexMul(Complex a, Complex b)
{
  Complex c;
  c.x = a.x * b.x - a.y * b.y;
  c.y = a.x * b.y + a.y * b.x;
  return c;
}

// Complex pointwise multiplication
static __global__ void ComplexPointwiseMulAndScale(Complex *a, const Complex *b,
  int size, float scale)
{
  const int numThreads = blockDim.x * gridDim.x;
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  
  for (int i = threadID; i < size; i += numThreads)
  {
    a[i] = ComplexScale(ComplexMul(a[i], b[i]), scale);
  }
}

void logLik_fft(double *y_, double *yhat_, int *ny, int *ns, double *h_, 
  double *m_, int *p, int *nThreads, double *out)
{
  
  arma::vec y    = getVec(y_, ny); // Convert to arma vec
  arma::vec yhat = getVec(yhat_, ns);
  double h;
  if (*h_==0) { h = bwNRD0(yhat, *m_); } else { h = (*m_)*(*h_); } 
  double z0   = std::min(y.min(), yhat.min()) - 3 * h;
  double z1   = std::max(y.max(), yhat.max()) + 3 * h;
  arma::vec z = arma::linspace<arma::vec>(z0, z1, std::pow(2, *p)) ;
  
  arma::vec binEdges  = getEdges(z) ;
  arma::vec filter    = getFilter(z0, z1, h, *p) ;  // Gauss filter
  double dt           = z[1] - z[0] ;
  arma::vec signal    = density(yhat, binEdges, dt) ;
  int nSignal         = signal.n_elem;
  int nFilter         = filter.n_elem;
  if (nSignal != nFilter) {fprintf(stderr, "signal & filter have unequal sizes\n");}
  
  int nblocks  = (*ns)/(*nThreads) + 1;   // Define the execution configuration
  
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
  
  // Transform signal to frequency domain
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
    PDF_smoothed[i] = h_out[i].x;
  }
  
  arma::vec PDF;   // Interpolate the grid likelihood to the data
  arma::interp1(z, PDF_smoothed, y, PDF);
  // arma::vec PDF_tmp = pmax(PDF, std::pow(10, -10)) ;
  *out = arma::accu(arma::log(pmax(PDF, std::pow(10, -10))));
  
  // cleanup memory
  free(h_signal);
  free(h_filter);
  free(h_out);
  cudaFree(d_signal);
  cudaFree(d_filter);
}


