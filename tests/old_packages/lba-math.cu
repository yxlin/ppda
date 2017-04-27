#include <R.h>
extern "C" void fptcdf(double *z, double *x0max, double *chi, double *driftrate, double *sddrift, double *out, int *n);
extern "C" void fptpdf(double *z, double *x0max, double *chi, double *driftrate, double *sddrift, double *out, int *n);
extern "C" void gpu2afc(double *z, double *x0max1, double *x0max2, double *chi1, double *chi2, double *driftrate1, double *driftrate2, double *sddrift1, double *sddrift2, double *out, int *n);
extern "C" void matrixgpu2afc(double *x, double *out, int *n);
extern "C" void floatmatgpu2afc(float *x, int *n);
extern "C" void lowmemgpu2afc(double *rt, int *cell, double *pars, int *map, int *nrts, int *npars, int *nchains, int *ncells, double *contaminants, double *out, int *dologsums);
extern "C" void docollogsum(double *in, int *nrow, int *ncol, double *out);


__global__ void
gpufptcdf(double *z, double *x0max, double *chi, double *driftrate, double *sddrift, double *out, int n) 
{
  double tmp1,tmp2,zs,zu,chiminuszu,xx,chizu,chizumax,dnormchizu,dnormchizumax;
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i < n)
  {
    if (x0max[i]<1e-10) {
      out[i]=((double) 1.0)-normcdf(((chi[i]/z[i])-driftrate[i])/sddrift[i]); /* LATER */
    } else {
      zs=z[i]*sddrift[i];
      zu=z[i]*driftrate[i];
      chiminuszu=chi[i]-zu ;
      xx=chiminuszu-x0max[i];
      chizu=chiminuszu/zs;
      chizumax=xx/zs;
      dnormchizumax= exp(-0.5*chizumax*chizumax - ((double) 0.91893853320467266954));
      dnormchizu = exp(-0.5*chizu*chizu - ((double) 0.91893853320467266954));
      tmp1=zs*(dnormchizumax-dnormchizu);
      tmp2=xx*normcdf(chizumax)-chiminuszu*normcdf(chizu);
      out[i]=((double) 1.0) + (tmp1+tmp2)/x0max[i];
    }
  }
}
  
void fptcdf(double *z, double *x0max, double *chi, double *driftrate, double *sddrift, double *out, int *n) 
{
  // Device Memory
  double *d_z, *d_x0max, *d_chi, *d_driftrate, *d_sddrift, *d_out;
  int nthreads,nblocks;
  // Define the execution configuration
  nthreads=128;
  nblocks=(*n)/nthreads + 1;
  // Allocate device arrays
  cudaMalloc((void**)&d_z, *n * sizeof(double));
  cudaMalloc((void**)&d_x0max, *n * sizeof(double));
  cudaMalloc((void**)&d_chi, *n * sizeof(double));
  cudaMalloc((void**)&d_driftrate, *n * sizeof(double));
  cudaMalloc((void**)&d_sddrift, *n * sizeof(double));
  cudaMalloc((void**)&d_out, *n * sizeof(double));
  // copy data to device
  cudaMemcpy(d_z, z, *n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x0max, x0max, *n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_chi, chi, *n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_driftrate, driftrate, *n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sddrift, sddrift, *n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, out, *n * sizeof(double), cudaMemcpyHostToDevice);
  // GPU fptcdf
  gpufptcdf<<<nblocks,nthreads>>>(d_z, d_x0max, d_chi, d_driftrate, d_sddrift, d_out, *n);
  // Copy output
  cudaMemcpy(out, d_out, *n * sizeof(double), cudaMemcpyDeviceToHost);
  //Rprintf("%f\n",out[0]);
  cudaFree(d_z);
  cudaFree(d_x0max);
  cudaFree(d_chi);
  cudaFree(d_driftrate);
  cudaFree(d_sddrift);
  cudaFree(d_out);
}


__global__ void
gpufptpdf(double *z, double *x0max, double *chi, double *driftrate, double *sddrift, double *out, int n) 
{
  double tmp1,tmp2,zs,zu,chiminuszu,xx,chizu,chizumax,dnormchizu,dnormchizumax;
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i < n)
  {
    if (x0max[i]<1e-10) {
      chizu=chi[i]/z[i] ; 
      xx=(chizu-driftrate[i])/sddrift[i];
      out[i]=(chizu/(z[i]*sddrift[i]))*exp(-0.5*xx*xx - ((double) 0.91893853320467266954));
    } else {
      zs=z[i]*sddrift[i];
      zu=z[i]*driftrate[i];
      chiminuszu=chi[i]-zu ;
      xx=chiminuszu-x0max[i];
      chizu=chiminuszu/zs;
      chizumax=xx/zs;
      dnormchizumax= exp(-0.5*chizumax*chizumax - ((double) 0.91893853320467266954));
      dnormchizu = exp(-0.5*chizu*chizu - ((double) 0.91893853320467266954));
      tmp1=sddrift[i]*(dnormchizumax-dnormchizu);
      tmp2=driftrate[i]*(normcdf(chizu)-normcdf(chizumax));
      out[i]=(tmp2+tmp1)/x0max[i];
    }
  }
}
  
void fptpdf(double *z, double *x0max, double *chi, double *driftrate, double *sddrift, double *out, int *n) 
{
  // Device Memory
  double *d_z, *d_x0max, *d_chi, *d_driftrate, *d_sddrift, *d_out;
  int nthreads,nblocks;
  // Define the execution configuration
  nthreads=128;
  nblocks=(*n)/nthreads + 1;
  // Allocate device arrays
  cudaMalloc((void**)&d_z, *n * sizeof(double));
  cudaMalloc((void**)&d_x0max, *n * sizeof(double));
  cudaMalloc((void**)&d_chi, *n * sizeof(double));
  cudaMalloc((void**)&d_driftrate, *n * sizeof(double));
  cudaMalloc((void**)&d_sddrift, *n * sizeof(double));
  cudaMalloc((void**)&d_out, *n * sizeof(double));
  // copy data to device
  cudaMemcpy(d_z, z, *n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x0max, x0max, *n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_chi, chi, *n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_driftrate, driftrate, *n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sddrift, sddrift, *n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, out, *n * sizeof(double), cudaMemcpyHostToDevice);
  // GPU fptpdf
  gpufptpdf<<<nblocks,nthreads>>>(d_z, d_x0max, d_chi, d_driftrate, d_sddrift, d_out, *n);
  // Copy output
  cudaMemcpy(out, d_out, *n * sizeof(double), cudaMemcpyDeviceToHost);
  //Rprintf("%f\n",out[0]);
  cudaFree(d_z);
  cudaFree(d_x0max);
  cudaFree(d_chi);
  cudaFree(d_driftrate);
  cudaFree(d_sddrift);
  cudaFree(d_out);
}


__global__ void
gpugpu2afc(double *z, double *x0max1, double *x0max2, double *chi1, double *chi2, double *driftrate1, double *driftrate2, double *sddrift1, double *sddrift2, double *out, int n) 
{
  double f1,F2,tmp1,tmp2,zs,zu,chiminuszu,xx,chizu,chizumax,dnormchizu,dnormchizumax;
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i < n)
  {
    // Calculate pdf on node1.
    if (x0max1[i]<1e-10) {
      chizu=chi1[i]/z[i] ; 
      xx=(chizu-driftrate1[i])/sddrift1[i];
      f1=(chizu/(z[i]*sddrift1[i]))*exp(-0.5*xx*xx - ((double) 0.91893853320467266954));
    } else {
      zs=z[i]*sddrift1[i];
      zu=z[i]*driftrate1[i];
      chiminuszu=chi1[i]-zu ;
      xx=chiminuszu-x0max1[i];
      chizu=chiminuszu/zs;
      chizumax=xx/zs;
      dnormchizumax= exp(-0.5*chizumax*chizumax - ((double) 0.91893853320467266954));
      dnormchizu = exp(-0.5*chizu*chizu - ((double) 0.91893853320467266954));
      tmp1=sddrift1[i]*(dnormchizumax-dnormchizu);
      tmp2=driftrate1[i]*(normcdf(chizu)-normcdf(chizumax));
      f1=(tmp2+tmp1)/x0max1[i];
    }
    // Calculate 1-cdf on node2.
    if (x0max2[i]<1e-10) {
      F2=((double) 1.0)-normcdf(((chi2[i]/z[i])-driftrate2[i])/sddrift2[i]); /* LATER */
    } else { 
      zs=z[i]*sddrift2[i];
      zu=z[i]*driftrate2[i];
      chiminuszu=chi2[i]-zu ;
      xx=chiminuszu-x0max2[i];
      chizu=chiminuszu/zs;
      chizumax=xx/zs;
      dnormchizumax= exp(-0.5*chizumax*chizumax - ((double) 0.91893853320467266954));
      dnormchizu = exp(-0.5*chizu*chizu - ((double) 0.91893853320467266954));
      tmp1=zs*(dnormchizumax-dnormchizu);
      tmp2=xx*normcdf(chizumax)-chiminuszu*normcdf(chizu);
      F2=  (-tmp1-tmp2)/x0max2[i];
    }
    out[i]=f1*F2;
  }
}

void gpu2afc(double *z, double *x0max1, double *x0max2, double *chi1, double *chi2, double *driftrate1, double *driftrate2, double *sddrift1, double *sddrift2, double *out, int *n)
{
  // Device Memory
  double *d_z, *d_x0max1, *d_x0max2, *d_chi1, *d_chi2, *d_driftrate1, *d_driftrate2, *d_sddrift1, *d_sddrift2, *d_out;
  int nthreads,nblocks;
  // Define the execution configuration
  nthreads=128;
  nblocks=(*n)/nthreads + 1;
  // Allocate device arrays
  cudaMalloc((void**)&d_z, *n * sizeof(double));
  cudaMalloc((void**)&d_x0max1, *n * sizeof(double));
  cudaMalloc((void**)&d_chi1, *n * sizeof(double));
  cudaMalloc((void**)&d_driftrate1, *n * sizeof(double));
  cudaMalloc((void**)&d_sddrift1, *n * sizeof(double));
  cudaMalloc((void**)&d_x0max2, *n * sizeof(double));
  cudaMalloc((void**)&d_chi2, *n * sizeof(double));
  cudaMalloc((void**)&d_driftrate2, *n * sizeof(double));
  cudaMalloc((void**)&d_sddrift2, *n * sizeof(double));
  cudaMalloc((void**)&d_out, *n * sizeof(double));
  // copy data to device
  cudaMemcpy(d_z, z, *n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x0max1, x0max1, *n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_chi1, chi1, *n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_driftrate1, driftrate1, *n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sddrift1, sddrift1, *n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x0max2, x0max2, *n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_chi2, chi2, *n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_driftrate2, driftrate2, *n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sddrift2, sddrift2, *n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, out, *n * sizeof(double), cudaMemcpyHostToDevice);
  // GPU gpu2afc
  gpugpu2afc<<<nblocks,nthreads>>>(d_z, d_x0max1, d_x0max2, d_chi1, d_chi2, d_driftrate1, d_driftrate2, d_sddrift1, d_sddrift2, d_out, *n);
  // Copy output
  cudaMemcpy(out, d_out, *n * sizeof(double), cudaMemcpyDeviceToHost);
  //Rprintf("%f\n",out[0]);
  cudaFree(d_z);
  cudaFree(d_x0max1);
  cudaFree(d_chi1);
  cudaFree(d_driftrate1);
  cudaFree(d_sddrift1);
  cudaFree(d_x0max2);
  cudaFree(d_chi2);
  cudaFree(d_driftrate2);
  cudaFree(d_sddrift2);
  cudaFree(d_out);
}

__global__ void
gpumatrixgpu2afc(double *x, double *out, int n) 
{
  double f1,F2,tmp1,tmp2,zs,zu,chiminuszu,xx,chizu,chizumax,dnormchizu,dnormchizumax;
  double z,x0max1,x0max2,chi1,chi2,driftrate1,driftrate2,sddrift1,sddrift2;
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int ii = 9*i; // memory location in array.
  const double dnorm = 0.91893853320467266954;
  if(i < n)
  {
    z=x[ii];
    x0max1=x[ii+1];
    x0max2=x[ii+2];
    chi1=x[ii+3];
    chi2=x[ii+4];
    driftrate1=x[ii+5];
    driftrate2=x[ii+6];
    sddrift1=x[ii+7];
    sddrift2=x[ii+8];
    
    // Calculate pdf on node1.
    if (x0max1<1e-10) {
      chizu=chi1/z ; 
      xx=(chizu-driftrate1)/sddrift1;
      f1=(chizu/(z*sddrift1))*exp(-0.5*xx*xx - dnorm);
    } else {
      zs=z*sddrift1;
      zu=z*driftrate1;
      chiminuszu=chi1-zu ;
      xx=chiminuszu-x0max1;
      chizu=chiminuszu/zs;
      chizumax=xx/zs;
      dnormchizumax= exp(-0.5*chizumax*chizumax - dnorm);
      dnormchizu = exp(-0.5*chizu*chizu - dnorm);
      tmp1=sddrift1*(dnormchizumax-dnormchizu);
      tmp2=driftrate1*(normcdf(chizu)-normcdf(chizumax));
      f1=(tmp2+tmp1)/x0max1;
    }
    // Calculate 1-cdf on node2.
    if (x0max2<1e-10) {
      F2=((double) 1.0)-normcdf(((chi2/z)-driftrate2)/sddrift2); /* LATER */
    } else { 
      zs=z*sddrift2;
      zu=z*driftrate2;
      chiminuszu=chi2-zu ;
      xx=chiminuszu-x0max2;
      chizu=chiminuszu/zs;
      chizumax=xx/zs;
      dnormchizumax= exp(-0.5*chizumax*chizumax - dnorm);
      dnormchizu = exp(-0.5*chizu*chizu - dnorm);
      tmp1=zs*(dnormchizumax-dnormchizu);
      tmp2=xx*normcdf(chizumax)-chiminuszu*normcdf(chizu);
      F2=  (-tmp1-tmp2)/x0max2;
    }
    out[i]=f1*F2;
  }
}


void matrixgpu2afc(double *x, double *out, int *n)
{ 
  // All inputs, and the output vector, are stacked together in matrix x. This
  // should have size 9 x *n. The order of the 1..10 is same as for the function
  // above, namely: z, A1, A2, b1, b2, v1, v2, s1, s2. Outputs are written over
  // the top of the first row (z).
  double *d_x, *d_out;
  int nthreads,nblocks,nmem;

  // These next three don't seem to help much past default compiler guesses.
  //cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
  //cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  //cudaDeviceSetCacheConfig(cudaFuncCachePreferShared); 
  nthreads=128;
  nblocks=(*n)/nthreads + 1;
  nmem = (*n)*9 ; // Size of the full array.
  // Allocate device arrays
  cudaMalloc((void**)&d_x, nmem * sizeof(double));
  cudaMemcpy(d_x, x, nmem * sizeof(double), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_out, (*n) * sizeof(double));
  // GPU kernel
  gpumatrixgpu2afc<<<nblocks,nthreads>>>(d_x, d_out, *n);
  // Copy output
  cudaMemcpy(out, d_out, (*n) * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_x);
  cudaFree(d_out);
}



__global__ void
gpufloatmatgpu2afc(float *x, int n) 
{
  float f1,F2,tmp1,tmp2,zs,zu,chiminuszu,xx,chizu,chizumax,dnormchizu,dnormchizumax;
  float z,x0max1,x0max2,chi1,chi2,driftrate1,driftrate2,sddrift1,sddrift2;
  const float  dnorm = 0.918938533;
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int ii = 9*i; // memory location in array.
  if(i < n)
  {
    z=x[ii];
    x0max1=x[ii+1];
    x0max2=x[ii+2];
    chi1=x[ii+3];
    chi2=x[ii+4];
    driftrate1=x[ii+5];
    driftrate2=x[ii+6];
    sddrift1=x[ii+7];
    sddrift2=x[ii+8];
    
    // Calculate pdf on node1.
    if (x0max1<1e-10) {
      chizu=chi1/z ; 
      xx=(chizu-driftrate1)/sddrift1;
      f1=(chizu/(z*sddrift1))*expf(-0.5*xx*xx - dnorm);
    } else {
      zs=z*sddrift1;
      zu=z*driftrate1;
      chiminuszu=chi1-zu ;
      xx=chiminuszu-x0max1;
      chizu=chiminuszu/zs;
      chizumax=xx/zs;
      dnormchizumax= expf(-0.5*chizumax*chizumax - dnorm);
      dnormchizu = expf(-0.5*chizu*chizu - dnorm);
      tmp1=sddrift1*(dnormchizumax-dnormchizu);
      tmp2=driftrate1*(normcdff(chizu)-normcdff(chizumax));
      f1=(tmp2+tmp1)/x0max1;
    }
    // Calculate 1-cdf on node2.
    if (x0max2<1e-10) {
      F2=((float) 1.0)-normcdff(((chi2/z)-driftrate2)/sddrift2); /* LATER */
    } else { 
      zs=z*sddrift2;
      zu=z*driftrate2;
      chiminuszu=chi2-zu ;
      xx=chiminuszu-x0max2;
      chizu=chiminuszu/zs;
      chizumax=xx/zs;
      dnormchizumax= expf(-0.5*chizumax*chizumax - dnorm);
      dnormchizu = expf(-0.5*chizu*chizu - dnorm);
      tmp1=zs*(dnormchizumax-dnormchizu);
      tmp2=xx*normcdff(chizumax)-chiminuszu*normcdff(chizu);
      F2=  (-tmp1-tmp2)/x0max2;
    }
    x[ii]=f1*F2;
  }
}

void floatmatgpu2afc(float *x, int *n)
{ 
  // All inputs, and the output vector, are stacked together in matrix x. This
  // should have size 9 x *n. The order of the 1..10 is same as for the function
  // above, namely: z, A1, A2, b1, b2, v1, v2, s1, s2. Output written over row1.
  float *d_x;
  int nthreads,nblocks,nmem;

  nthreads=128;
  nblocks=(*n)/nthreads + 1;
  nmem = (*n)*9 ; // Size of the full array.
  // Allocate device arrays
  cudaMalloc((void**)&d_x, nmem * sizeof(float));
  cudaMemcpy(d_x, x, nmem * sizeof(float), cudaMemcpyHostToDevice);
  // GPU kernel
  gpufloatmatgpu2afc<<<nblocks,nthreads>>>(d_x, *n);
  // Copy output
  cudaMemcpy(x, d_x, nmem * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_x);
}

__global__ void collogsum(double *x, int nrow, int ncol, double *out) 
{
  int i;
  double y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;  
  int j = col*nrow;

  y=(double) 0.0;
  if (col < ncol) {
    for (i=0;i<nrow;i++) {
      y=y+log(x[i + j]);
    }
    out[col]=y;
  }
}

void docollogsum(double *in, int *nrow, int *ncol, double *out) 
{
  double *d_in,*d_out;
  int nthreads=32;
  int nblocks = (*ncol)/nthreads +1;

  cudaMalloc((void**)&d_in, (*nrow)*(*ncol)*sizeof(double));
  cudaMalloc((void**)&d_out, (*ncol)*sizeof(double));
  // Copy data to device.
  cudaMemcpy(d_in, in, (*nrow)*(*ncol)*sizeof(double), cudaMemcpyHostToDevice);
  // Call the kernel.
  collogsum<<<nblocks,nthreads>>>(d_in,*nrow,*ncol,d_out);
  // Copy output back to host.
  cudaMemcpy(out, d_out, (*ncol)*sizeof(double), cudaMemcpyDeviceToHost);
  // Free memory.
  cudaFree(d_in);
  cudaFree(d_out);
}

__global__ void  gpulowmemgpu2afc(double *rt, int *cell, double *pars, int *map,
    int nrts, int npars, int nchains, int ncells, double *contaminants, double *out)
{
  double f1,F2,tmp1,tmp2,zs,zu,chiminuszu,xx,chizu,chizumax,dnormchizu,dnormchizumax;
  double basert,z,x0max1,x0max2,chi1,chi2,driftrate1,driftrate2,sddrift1,sddrift2;
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  //int chain,celloffset,chainoffset,rtoffset,t0offset,A1offset,A2offset,b1offset;
  int chain,celloffset,chainoffset,t0offset,A1offset,A2offset,b1offset;
  int b2offset,v1offset,v2offset,s1offset,s2offset;
  const double dnorm = 0.91893853320467266954;

  if(i < nrts)
  {
    // Get a bunch of things from arrays just once, to save mem access later.
    basert = rt[i];
    celloffset = 9*(cell[i]-1); // Get the right column of mapping for this RT's cell. 
    t0offset = map[celloffset]-1; // t0 in first row.
    A1offset = map[celloffset+1]-1; // x0max1.
    A2offset = map[celloffset+2]-1; // x0max2.
    b1offset = map[celloffset+3]-1; // chi1.
    b2offset = map[celloffset+4]-1; // chi2.
    v1offset = map[celloffset+5]-1; // drift1.
    v2offset = map[celloffset+6]-1; // drift2.
    s1offset = map[celloffset+7]-1; // sddrift1.
    s2offset = map[celloffset+8]-1; // sddrift2.

//    rtoffset = i*nchains;
    for (chain=0;chain<nchains;chain++) {
      chainoffset=chain*npars;
      z=basert-pars[t0offset+chainoffset];
      if (z<1e-10) {
        z=(double) 0.0;
      }
      x0max1 = pars[A1offset+chainoffset];
      x0max2 = pars[A2offset+chainoffset];
      chi1   = pars[b1offset+chainoffset];
      chi2 =   pars[b2offset+chainoffset];
      driftrate1 = pars[v1offset+chainoffset];
      driftrate2 = pars[v2offset+chainoffset];
      sddrift1 = pars[s1offset+chainoffset];
      sddrift2 = pars[s2offset+chainoffset];
    
      // Calculate pdf on node1.
      if (x0max1<1e-10) {
        chizu=chi1/z ; 
        xx=(chizu-driftrate1)/sddrift1;
        f1=(chizu/(z*sddrift1))*exp(-0.5*xx*xx - dnorm);
      } else {
        zs=z*sddrift1;
        zu=z*driftrate1;
        chiminuszu=chi1-zu ;
        xx=chiminuszu-x0max1;
        chizu=chiminuszu/zs;
        chizumax=xx/zs;
        dnormchizumax= exp(-0.5*chizumax*chizumax - dnorm);
        dnormchizu = exp(-0.5*chizu*chizu - dnorm);
        tmp1=sddrift1*(dnormchizumax-dnormchizu);
        tmp2=driftrate1*(normcdf(chizu)-normcdf(chizumax));
        f1=(tmp2+tmp1)/x0max1;
      }
      // Calculate 1-cdf on node2.
      if (x0max2<1e-10) {
        F2=((double) 1.0)-normcdf(((chi2/z)-driftrate2)/sddrift2); /* LATER */
      } else { 
        zs=z*sddrift2;
        zu=z*driftrate2;
        chiminuszu=chi2-zu ;
        xx=chiminuszu-x0max2;
        chizu=chiminuszu/zs;
        chizumax=xx/zs;
        dnormchizumax= exp(-0.5*chizumax*chizumax - dnorm);
        dnormchizu = exp(-0.5*chizu*chizu - dnorm);
        tmp1=zs*(dnormchizumax-dnormchizu);
        tmp2=xx*normcdf(chizumax)-chiminuszu*normcdf(chizu);
        F2=  (-tmp1-tmp2)/x0max2;
      }
//      out[rtoffset+chain]=(((double) 1.0) -contaminants[0])*f1*F2 + ((double) 0.5)*contaminants[1]*contaminants[0];
      out[chain*nrts+i]=(((double) 1.0) -contaminants[0])*f1*F2 + ((double) 0.5)*contaminants[1]*contaminants[0];

    }

  }
}


void lowmemgpu2afc(double *rt, int *cell, double *pars, int *map, int *nrts, int *npars,
     int *nchains, int *ncells, double *contaminants, double *out, int *dologsums) 
{
  // rt and cell are vectors of nrts length. pars is an array of npars rows
  // by nchains columns. For every combination of chain and RT, this function 
  // fills in the density. Uses map to extract the right parameters. map must be
  // an integer matrix of ncell columns, with 9 rows. Each entry gives an index
  // of where to find a parameter for the corresponding cell to that column. The
  // rows give entries for (in this order!): t0, A1, A2, b1, b2, v1, v2, s1, s2.
  // Any entry in *map that is larger than npars will seg fault. Contaminants
  // should be a 2-vector, with first the probability of a contaminant, and then
  // the density.  Output is in  out, which is an array of nchains rows by nrt 
  // columns (if dologsums==0) or a vector of nchains length (otherwise).
  double *d_rt, *d_pars, *d_out, *d_contaminants, *d_smallout;
  int nthreads, nblocks, *d_cell, *d_map;
//Rprintf("%f %f %f %d %f %d %d %d %d\n",rt[0],contaminants[0],contaminants[1],cell[0],pars[0],map[0],*nrts,*npars,*nchains);

  nthreads=128;
  nblocks=(*nrts)/nthreads + 1;
  // Allocate device arrays.
  cudaMalloc((void**)&d_rt, (*nrts)*sizeof(double));
  cudaMalloc((void**)&d_pars, (*npars)*(*nchains)*sizeof(double));
  cudaMalloc((void**)&d_out, (*nrts)*(*nchains)*sizeof(double));
  cudaMalloc((void**)&d_cell, (*nrts)*sizeof(int));
  cudaMalloc((void**)&d_map, (9)*(*ncells)*sizeof(int));
  cudaMalloc((void**)&d_contaminants, 2*sizeof(double));
  // Copy data to device.
  cudaMemcpy(d_rt, rt, (*nrts)*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_pars, pars, (*npars)*(*nchains)*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_cell, cell, (*nrts)*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_map, map, (9)*(*ncells)*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_contaminants, contaminants, 2*sizeof(double), cudaMemcpyHostToDevice);
  // Call the kernel.
  gpulowmemgpu2afc<<<nblocks,nthreads>>>(d_rt, d_cell, d_pars, d_map, *nrts, *npars, *nchains, *ncells, d_contaminants, d_out);


  if ((*dologsums)==0) {
    // Copy full-sized output back to host.
    cudaMemcpy(out, d_out, (*nrts)*(*nchains)*sizeof(double), cudaMemcpyDeviceToHost);
  } else {
    // Calculate log-sum of each row and return that.
    cudaMalloc((void**)&d_smallout, (*nchains)*sizeof(double));
    nblocks=(*nchains)/nthreads + 1;
    collogsum<<<nblocks,nthreads>>>(d_out,*nrts,*nchains,d_smallout);
    // Copy reduced output back to host.
    cudaMemcpy(out, d_smallout, (*nchains)*sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_smallout);
  }
  // Free memory.
  cudaFree(d_rt);
  cudaFree(d_pars);
  cudaFree(d_out);
  cudaFree(d_cell);
  cudaFree(d_map);
  cudaFree(d_contaminants);
}
