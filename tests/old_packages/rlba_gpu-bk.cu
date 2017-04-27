__global__ void rlba(unsigned int seed, curandState_t* state, int n, double* b,
                     double* A, double* v1, double* v2, double* sv,
                     double* t0, double* RT, int* R) {
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
    double x01, x02, rate1, rate2, dt1, dt2;

    // Algorithm 1; 'expl'; use when lower > mean; upper = INFINITY
    // rtnorm1 setting
    int valid = 0;              // alphaStar  must be greater than 0
    double z = 0, r = 0, u = 0; // a stands for alphaStar in Robert (1995)
    double stdl1, stdu1, stdl2, stdu2, a1, a2, valid_z;
    stdl1 = (0 - v1[0]) / sv[0];
    stdu1 = (CUDART_INF - v1[0]) / sv[0];
    a1 = 0.5 * (sqrt(stdl1 * stdl1 + 4.0) + stdl1);

    stdl2 = (0 - v2[0]) / sv[0];
    stdu2 = (CUDART_INF - v2[0]) / sv[0];
    a2 = 0.5 * (sqrt(stdl2 * stdl2 + 4.0) + stdl2);

    for (int i = threadID; i < n; i += numThreads)
    {
        curand_init(seed, i, 0, &state[i]);
        x01 = A[0] * curand_uniform_double(&state[i]);
        x02 = A[0] * curand_uniform_double(&state[i]);

        rate1 = rtnorm1_device(i, &state[i], a1, v1[0], sv[0], stdl1, stdu1);
        rate2 = rtnorm1_device(i, &state[i], a2, v2[0], sv[0], stdl2, stdu2);

        /*
        valid = 0;
        while (valid == 0) {
           z = (-1/a1) * log(curand_uniform_double(&state[i])) + stdl1;
           u = curand_uniform_double(&state[i]);
           r = exp(-0.5 * (z - a1) * (z - a1));
           if (u <= r && z <= stdu1)
           {
             valid_z = z;
             valid   = 1;
           }
        }
        rate1 = v1[0] + sv[0] * valid_z;
        
        valid = 0;
        while (valid == 0) {
           z = (-1/a2) * log(curand_uniform_double(&state[i])) + stdl2;
           u = curand_uniform_double(&state[i]);
           r = exp(-0.5 * (z - a2) * (z - a2));
           if (u <= r && z <= stdu2)
           {
             valid_z = z;
             valid   = 1;
           }
        }
        rate2 = v2[0] + sv[0] * valid_z;
        */

        printf("Trial %d has drift rates: [%.4f %.4f]\n", i, rate1, rate2);

        dt1 = (b[0] - x01) / rate1;
        dt2 = (b[0] - x02) / rate2;
        RT[i] = (dt1 < dt2) ? (dt1 + t0[0]) : (dt2 + t0[0]);
        R[i]  = (dt1 < dt2) ? 1 : 2;

        // printf("Trial %d has starting points and rates: %.2f, %.2f, %.2f, %.2f\n", i, x01, x02,
        //               rate1, rate2);
        // printf("Trial %d has choice time 1: %.2f, and choice time 2: %.2f\n", i, dt1, dt2);
    }
    
}

