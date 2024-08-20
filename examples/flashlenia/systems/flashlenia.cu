#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <cstdio>
#include <cmath>
#include <vector>

const int NUM_SIMULATIONS = 1000;
const int GRID_SIZE = 128;
const int BLOCK_SIZE = 8;
const float DT = 0.1f;
const int MAX_ITERATIONS = 1000;
const float THRESHOLD = 0.01f;

__constant__ float kernel[9];

// nvcc flashlenia.cu -o flashlenia

// CUDA error checking macro
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

__device__ float growth(float u) {
    return (u > 0.5f) ? 1.0f : -1.0f;
}

__global__ void lenia_step(float* state, float* next_state, int width, int height, int* is_active) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y;

    for (int y = idy; y < height; y += stride_y) {
        for (int x = idx; x < width; x += stride_x) {
            int index = y * width + x;
            float sum = 0.0f;

            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int nx = (x + dx + width) % width;
                    int ny = (y + dy + height) % height;
                    sum += state[ny * width + nx] * kernel[(dy + 1) * 3 + (dx + 1)];
                }
            }

            float u = state[index];
            float du = growth(sum) * DT;
            next_state[index] = fmaxf(0.0f, fminf(1.0f, u + du));

            // Check activity
            if (next_state[index] > THRESHOLD) {
                *is_active = 1;
            }
        }
    }
}

__global__ void init_random_state(float* state, int size, curandState* rand_states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size; i += stride) {
        state[i] = curand_uniform(&rand_states[i]);
    }
}

__global__ void setup_rand_states(curandState* rand_states, int size, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size; i += stride) {
        curand_init(seed, i, 0, &rand_states[i]);
    }
}

__global__ void compute_entropy(float* state, float* entropy, int size) {
    __shared__ float local_sum[256];
    __shared__ float total_sum;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float sum = 0.0f;
    float normalizing_sum = 0.0f;

    // First pass: calculate the sum for normalization
    for (int i = idx; i < size; i += stride) {
        normalizing_sum += state[i];
    }

    // Reduce normalizing_sum
    local_sum[threadIdx.x] = normalizing_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            local_sum[threadIdx.x] += local_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        total_sum = local_sum[0];
    }
    __syncthreads();

    // Second pass: compute entropy with normalized probabilities
    for (int i = idx; i < size; i += stride) {
        float p = state[i] / total_sum;
        if (p > 1e-10f) {  // Avoid log(0) and very small values
            sum -= p * __log2f(p);
        }
    }

    // Reduce entropy sum
    local_sum[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            local_sum[threadIdx.x] += local_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(entropy, local_sum[0] / size);  // Divide by number of cells
    }
}

void run_simulation(float* d_state, float* d_next_state, int width, int height, int* d_is_active, int* iterations_run, int* h_is_active, float* d_entropy, float* h_entropy) {
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    *h_is_active = 1;
    int iter;
    for (iter = 0; iter < MAX_ITERATIONS && *h_is_active; iter++) {
        lenia_step<<<grid_size, block_size>>>(d_state, d_next_state, width, height, d_is_active);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        CHECK_CUDA_ERROR(cudaMemcpy(h_is_active, d_is_active, sizeof(int), cudaMemcpyDeviceToHost));
   //     CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        if (*h_is_active == 0) {
            break;  // No activity detected, stop simulation early
        }

        // Reset activity state for next iteration
        CHECK_CUDA_ERROR(cudaMemset(d_is_active, 0, sizeof(int)));

        std::swap(d_state, d_next_state);
    }
    *iterations_run = iter;
    
    // Compute entropy on the last state
    CHECK_CUDA_ERROR(cudaMemset(d_entropy, 0, sizeof(float)));
    compute_entropy<<<grid_size, block_size>>>(d_state, d_entropy, width * height);
    CHECK_CUDA_ERROR(cudaMemcpy(h_entropy, d_entropy, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

void print_state(float* state, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float value = state[y * width + x];
            if (value > 0.9f) {
                printf("█");
            } else if (value > 0.7f) {
                printf("▓");
            } else if (value > 0.5f) {
                printf("▒");
            } else if (value > 0.3f) {
                printf("░");
            } else {
                printf(" ");
            }
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    float h_kernel[9];
    
    if (argc == 10) {
        // If 9 arguments are provided (plus the program name), use them as h_kernel values
        for (int i = 0; i < 9; i++) {
            h_kernel[i] = std::stof(argv[i + 1]);
        }
    } else {
        // Use default values if 9 arguments are not provided
        float default_kernel[9] = {0.05f, 0.2f, 0.05f, 0.2f, 0.0f, 0.2f, 0.05f, 0.2f, 0.05f};
        std::copy(std::begin(default_kernel), std::end(default_kernel), h_kernel);
    }

    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(kernel, h_kernel, 9 * sizeof(float)));

    std::vector<float*> d_states(NUM_SIMULATIONS);
    std::vector<float*> d_next_states(NUM_SIMULATIONS);
    std::vector<int*> d_is_active(NUM_SIMULATIONS);  // Device active state array
    std::vector<float*> d_entropies(NUM_SIMULATIONS);
    std::vector<int> h_is_active(NUM_SIMULATIONS);  // Host active state array
    std::vector<float> h_entropies(NUM_SIMULATIONS); // Host entropy array

    size_t size = GRID_SIZE * GRID_SIZE * sizeof(float);

    for (int i = 0; i < NUM_SIMULATIONS; i++) {
        CHECK_CUDA_ERROR(cudaMalloc(&d_states[i], size));
        CHECK_CUDA_ERROR(cudaMalloc(&d_next_states[i], size));
        CHECK_CUDA_ERROR(cudaMalloc(&d_is_active[i], sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_entropies[i], sizeof(float)));
    }

    std::vector<int> iterations_run(NUM_SIMULATIONS);

    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size((GRID_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, (GRID_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Initialize random state
    curandState* init_rand_states;
    CHECK_CUDA_ERROR(cudaMalloc(&init_rand_states, GRID_SIZE * GRID_SIZE * sizeof(curandState)));

    setup_rand_states<<<grid_size, block_size>>>(init_rand_states, GRID_SIZE * GRID_SIZE, time(NULL));

    for (int i = 0; i < NUM_SIMULATIONS; i++) {
        init_random_state<<<grid_size, block_size>>>(d_states[i], GRID_SIZE * GRID_SIZE, init_rand_states);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    // Run each simulation sequentially
    for (int i = 0; i < NUM_SIMULATIONS; i++) {
        run_simulation(d_states[i], d_next_states[i], GRID_SIZE, GRID_SIZE, d_is_active[i], &iterations_run[i], &h_is_active[i], d_entropies[i], &h_entropies[i]);
    }

    // Compute mean and variance of entropies
    float mean_entropy = 0.0f;
    for (float entropy : h_entropies) {
        mean_entropy += entropy;
    }
    mean_entropy /= NUM_SIMULATIONS;

    float variance_entropy = 0.0f;
    for (float entropy : h_entropies) {
        variance_entropy += (entropy - mean_entropy) * (entropy - mean_entropy);
    }
    variance_entropy /= (NUM_SIMULATIONS - 1);

    std::cout << "Mean entropy: " << mean_entropy << std::endl;
    std::cout << "Variance of entropy: " << variance_entropy << std::endl;

    // Clean up
    for (int i = 0; i < NUM_SIMULATIONS; i++) {
        CHECK_CUDA_ERROR(cudaFree(d_states[i]));
        CHECK_CUDA_ERROR(cudaFree(d_next_states[i]));
        CHECK_CUDA_ERROR(cudaFree(d_is_active[i]));
        CHECK_CUDA_ERROR(cudaFree(d_entropies[i]));
    }

    CHECK_CUDA_ERROR(cudaFree(init_rand_states));

    return 0;
}
