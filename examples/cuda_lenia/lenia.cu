#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <math.h>

#define WIDTH 128
#define HEIGHT 128
#define BLOCK_SIZE 16
#define KERNEL_RADIUS 13
#define KERNEL_SIZE (2 * KERNEL_RADIUS + 1)
#define DT 0.1f
#define NUM_KERNELS 3
#define NUM_INSTANCES 400

__constant__ float kernels[NUM_KERNELS][KERNEL_SIZE * KERNEL_SIZE];
__constant__ float growth_centers[NUM_KERNELS];
__constant__ float growth_widths[NUM_KERNELS];

__global__ void initialize(float* state, curandState* rand_states, unsigned long long seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) return;

    int idx = y * WIDTH + x;
    curand_init(seed, idx, 0, &rand_states[idx]);
    state[idx] = curand_uniform(&rand_states[idx]);
}

__device__ float growth(float x, float center, float width) {
    float dx = x - center;
    return 2.0f * expf(-dx * dx / (2.0f * width * width)) - 1.0f;
}

__global__ void update(float* state, float* new_state, float* total_activity) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) return;

    float sums[NUM_KERNELS] = {0.0f};
    for (int dy = -KERNEL_RADIUS; dy <= KERNEL_RADIUS; dy++) {
        for (int dx = -KERNEL_RADIUS; dx <= KERNEL_RADIUS; dx++) {
            int nx = (x + dx + WIDTH) % WIDTH;
            int ny = (y + dy + HEIGHT) % HEIGHT;
            int kidx = (dy + KERNEL_RADIUS) * KERNEL_SIZE + (dx + KERNEL_RADIUS);
            float cell_state = state[ny * WIDTH + nx];
            
            for (int k = 0; k < NUM_KERNELS; k++) {
                sums[k] += cell_state * kernels[k][kidx];
            }
        }
    }

    int idx = y * WIDTH + x;
    float new_value = state[idx];
    for (int k = 0; k < NUM_KERNELS; k++) {
        new_value += DT * growth(sums[k], growth_centers[k], growth_widths[k]);
    }
    new_state[idx] = fmaxf(0.0f, fminf(1.0f, new_value));

    // Calculate the absolute activity change and add to the total activity
    float activity_change = fabsf(new_state[idx] - state[idx]);
    atomicAdd(total_activity, activity_change);
}


__global__ void calculateMetrics(float* state, float* sum_activity, float* sum_entropy) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) return;

    int idx = y * WIDTH + x;
    float cell_state = state[idx];

    float activity = fabsf(cell_state - 0.5f) * 2.0f;
    atomicAdd(sum_activity, activity);

    float entropy = 0.0f;
    if (cell_state > 0.0f && cell_state < 1.0f) {
        entropy = -cell_state * logf(cell_state) - (1.0f - cell_state) * logf(1.0f - cell_state);
    }
    atomicAdd(sum_entropy, entropy);
}

void init_kernels(float* h_growth_centers, float* h_growth_widths) {
    float h_kernels[NUM_KERNELS][KERNEL_SIZE * KERNEL_SIZE];

    for (int k = 0; k < NUM_KERNELS; k++) {
        float sum = 0.0f;
        for (int y = 0; y < KERNEL_SIZE; y++) {
            for (int x = 0; x < KERNEL_SIZE; x++) {
                float dx = x - KERNEL_RADIUS;
                float dy = y - KERNEL_RADIUS;
                float r = sqrtf(dx * dx + dy * dy) / KERNEL_RADIUS;
                h_kernels[k][y * KERNEL_SIZE + x] = r < 1.0f ? powf(1.0f - r * r, k + 1) : 0.0f;
                sum += h_kernels[k][y * KERNEL_SIZE + x];
            }
        }
        for (int i = 0; i < KERNEL_SIZE * KERNEL_SIZE; i++) {
            h_kernels[k][i] /= sum;
        }
    }

    cudaMemcpyToSymbol(kernels, h_kernels, NUM_KERNELS * KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    cudaMemcpyToSymbol(growth_centers, h_growth_centers, NUM_KERNELS * sizeof(float));
    cudaMemcpyToSymbol(growth_widths, h_growth_widths, NUM_KERNELS * sizeof(float));
}

void plotASCII(float* state, const char* title) {
    printf("%s\n", title);
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            float value = state[y * WIDTH + x];
            if (value < 0.2f) printf(" ");
            else if (value < 0.4f) printf(".");
            else if (value < 0.6f) printf("o");
            else if (value < 0.8f) printf("O");
            else printf("#");
        }
        printf("\n");
    }
    printf("\n");
}

void calculateStatistics(float* metrics, int num_instances, float* mean, float* variance) {
    float sum = 0.0f;
    for (int i = 0; i < num_instances; i++) {
        sum += metrics[i];
    }
    *mean = sum / num_instances;

    float sum_sq_diff = 0.0f;
    for (int i = 0; i < num_instances; i++) {
        float diff = metrics[i] - *mean;
        sum_sq_diff += diff * diff;
    }
    *variance = sum_sq_diff / (num_instances - 1);
}

int main(int argc, char** argv) {
    if (argc != (2 * NUM_KERNELS) + 1) {
        fprintf(stderr, "Usage: %s <growth_center_1> <growth_width_1> ... <growth_center_%d> <growth_width_%d>\n", argv[0], NUM_KERNELS, NUM_KERNELS);
        return 1;
    }

    float h_growth_centers[NUM_KERNELS];
    float h_growth_widths[NUM_KERNELS];
    for (int i = 0; i < NUM_KERNELS; i++) {
        h_growth_centers[i] = atof(argv[1 + i * 2]);
        h_growth_widths[i] = atof(argv[2 + i * 2]);
    }

    float *d_state, *d_new_state, *d_sum_activity, *d_sum_entropy;
    float *h_state_initial, *h_state_final;
    curandState *d_rand_states;

    float activity_metrics[NUM_INSTANCES];
    float entropy_metrics[NUM_INSTANCES];

    cudaMalloc(&d_state, WIDTH * HEIGHT * sizeof(float));
    cudaMalloc(&d_new_state, WIDTH * HEIGHT * sizeof(float));
    cudaMalloc(&d_sum_activity, sizeof(float));
    cudaMalloc(&d_sum_entropy, sizeof(float));
    cudaMalloc(&d_rand_states, WIDTH * HEIGHT * sizeof(curandState));

    h_state_initial = (float*)malloc(WIDTH * HEIGHT * sizeof(float));
    h_state_final = (float*)malloc(WIDTH * HEIGHT * sizeof(float));

    init_kernels(h_growth_centers, h_growth_widths);

    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size((WIDTH + BLOCK_SIZE - 1) / BLOCK_SIZE, (HEIGHT + BLOCK_SIZE - 1) / BLOCK_SIZE);

    float* d_total_activity;
cudaMalloc(&d_total_activity, sizeof(float));

    for (int instance = 0; instance < NUM_INSTANCES; instance++) {
        unsigned long long seed = instance;
        initialize<<<grid_size, block_size>>>(d_state, d_rand_states, seed);

        cudaMemcpy(h_state_initial, d_state, WIDTH * HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);

        for (int i = 0; i < 300; i++) {
        cudaMemset(d_total_activity, 0, sizeof(float));

        update<<<grid_size, block_size>>>(d_state, d_new_state, d_total_activity);
        
                    std::swap(d_state, d_new_state);

                            float h_total_activity;
        cudaMemcpy(&h_total_activity, d_total_activity, sizeof(float), cudaMemcpyDeviceToHost);

        if (h_total_activity == 0.0f) {
            printf("Early stopping at iteration %d due to zero activity\n", i);
            break;
        }

        }

        cudaMemcpy(h_state_final, d_state, WIDTH * HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);

        cudaMemset(d_sum_activity, 0, sizeof(float));
        cudaMemset(d_sum_entropy, 0, sizeof(float));
        
        calculateMetrics<<<grid_size, block_size>>>(d_state, d_sum_activity, d_sum_entropy);
        
        float h_sum_activity, h_sum_entropy;
        cudaMemcpy(&h_sum_activity, d_sum_activity, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_sum_entropy, d_sum_entropy, sizeof(float), cudaMemcpyDeviceToHost);

        float avg_activity = h_sum_activity / (WIDTH * HEIGHT);
        float spatial_entropy = h_sum_entropy / (WIDTH * HEIGHT);

        activity_metrics[instance] = avg_activity;
        entropy_metrics[instance] = spatial_entropy;

     //   char title[50];
        // snprintf(title, sizeof(title), "Instance %d Initial State:", instance + 1);
        // plotASCII(h_state_initial, title);
     //   snprintf(title, sizeof(title), "Instance %d Final State:", instance + 1);
     //   plotASCII(h_state_final, title);
    }

    float mean_activity, variance_activity;
    float mean_entropy, variance_entropy;
    
    calculateStatistics(activity_metrics, NUM_INSTANCES, &mean_activity, &variance_activity);
    calculateStatistics(entropy_metrics, NUM_INSTANCES, &mean_entropy, &variance_entropy);

    printf("Final State Metrics:\n");
    printf("Average Activity (Mean): %.8f\n", mean_activity);
    printf("Average Activity (Variance): %.8f\n", variance_activity);
    printf("Spatial Entropy (Mean): %.8f\n", mean_entropy);
    printf("Spatial Entropy (Variance): %.8f\n", variance_entropy);

    cudaFree(d_state);
    cudaFree(d_new_state);
    cudaFree(d_sum_activity);
    cudaFree(d_sum_entropy);
    cudaFree(d_rand_states);
    free(h_state_initial);
    free(h_state_final);

    return 0;
}