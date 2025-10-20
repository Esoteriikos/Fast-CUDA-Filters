#include "utils.cuh"

/**
 * Box blur (mean filter) using shared memory
 * 
 * Optimized uniform averaging filter
 * Can be further optimized using integral images for large kernels
 */
__global__ void boxBlurKernel(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int width,
    int height,
    int radius
) {
    __shared__ float s_tile[TILE_HEIGHT + 2 * GAUSSIAN_KERNEL_RADIUS][TILE_WIDTH + 2 * GAUSSIAN_KERNEL_RADIUS];
    
    int global_x = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int global_y = blockIdx.y * TILE_HEIGHT + threadIdx.y;
    
    int s_x = threadIdx.x + radius;
    int s_y = threadIdx.y + radius;
    
    // Load center
    if (global_x < width && global_y < height) {
        s_tile[s_y][s_x] = (float)input[coordsToIndex(global_x, global_y, width)];
    } else {
        s_tile[s_y][s_x] = 0.0f;
    }
    
    // Load aprons (similar to Gaussian blur)
    if (threadIdx.x < radius) {
        int apron_x = max(0, global_x - radius);
        if (global_y < height) {
            s_tile[s_y][threadIdx.x] = (float)input[coordsToIndex(apron_x, global_y, width)];
        } else {
            s_tile[s_y][threadIdx.x] = 0.0f;
        }
    }
    
    if (threadIdx.x >= TILE_WIDTH - radius && threadIdx.x < TILE_WIDTH) {
        int apron_x = min(width - 1, global_x + radius);
        if (global_y < height && apron_x < width) {
            s_tile[s_y][s_x + radius] = (float)input[coordsToIndex(apron_x, global_y, width)];
        } else {
            s_tile[s_y][s_x + radius] = 0.0f;
        }
    }
    
    if (threadIdx.y < radius) {
        int apron_y = max(0, global_y - radius);
        if (global_x < width) {
            s_tile[threadIdx.y][s_x] = (float)input[coordsToIndex(global_x, apron_y, width)];
        } else {
            s_tile[threadIdx.y][s_x] = 0.0f;
        }
    }
    
    if (threadIdx.y >= TILE_HEIGHT - radius && threadIdx.y < TILE_HEIGHT) {
        int apron_y = min(height - 1, global_y + radius);
        if (global_x < width && apron_y < height) {
            s_tile[s_y + radius][s_x] = (float)input[coordsToIndex(global_x, apron_y, width)];
        } else {
            s_tile[s_y + radius][s_x] = 0.0f;
        }
    }
    
    __syncthreads();
    
    // Compute box blur
    if (global_x < width && global_y < height) {
        float sum = 0.0f;
        int count = 0;
        
        #pragma unroll
        for (int ky = -radius; ky <= radius; ky++) {
            #pragma unroll
            for (int kx = -radius; kx <= radius; kx++) {
                sum += s_tile[s_y + ky][s_x + kx];
                count++;
            }
        }
        
        float mean = sum / (float)count;
        output[coordsToIndex(global_x, global_y, width)] = (unsigned char)clampf(mean, 0.0f, 255.0f);
    }
}

/**
 * Separable box blur - horizontal pass
 * More efficient for large kernels
 */
__global__ void boxBlurHorizontal(
    const unsigned char* __restrict__ input,
    float* __restrict__ temp,
    int width,
    int height,
    int radius
) {
    __shared__ float s_row[TILE_WIDTH + 2 * GAUSSIAN_KERNEL_RADIUS];
    
    int global_x = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (global_y >= height) return;
    
    int s_x = threadIdx.x + radius;
    
    // Load data
    if (global_x < width) {
        s_row[s_x] = (float)input[coordsToIndex(global_x, global_y, width)];
    } else {
        s_row[s_x] = 0.0f;
    }
    
    if (threadIdx.x < radius) {
        int apron_x = max(0, global_x - radius);
        s_row[threadIdx.x] = (float)input[coordsToIndex(apron_x, global_y, width)];
    }
    
    if (threadIdx.x >= TILE_WIDTH - radius && threadIdx.x < TILE_WIDTH) {
        int apron_x = min(width - 1, global_x + radius);
        if (apron_x < width) {
            s_row[s_x + radius] = (float)input[coordsToIndex(apron_x, global_y, width)];
        }
    }
    
    __syncthreads();
    
    if (global_x < width) {
        float sum = 0.0f;
        
        #pragma unroll
        for (int k = -radius; k <= radius; k++) {
            sum += s_row[s_x + k];
        }
        
        int kernel_size = 2 * radius + 1;
        temp[coordsToIndex(global_x, global_y, width)] = sum / (float)kernel_size;
    }
}

/**
 * Separable box blur - vertical pass
 */
__global__ void boxBlurVertical(
    const float* __restrict__ temp,
    unsigned char* __restrict__ output,
    int width,
    int height,
    int radius
) {
    __shared__ float s_col[TILE_HEIGHT + 2 * GAUSSIAN_KERNEL_RADIUS][TILE_WIDTH];
    
    int global_x = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int global_y = blockIdx.y * TILE_HEIGHT + threadIdx.y;
    
    int s_y = threadIdx.y + radius;
    
    // Load data
    if (global_x < width && global_y < height) {
        s_col[s_y][threadIdx.x] = temp[coordsToIndex(global_x, global_y, width)];
    } else {
        s_col[s_y][threadIdx.x] = 0.0f;
    }
    
    if (threadIdx.y < radius) {
        int apron_y = max(0, global_y - radius);
        if (global_x < width) {
            s_col[threadIdx.y][threadIdx.x] = temp[coordsToIndex(global_x, apron_y, width)];
        }
    }
    
    if (threadIdx.y >= TILE_HEIGHT - radius && threadIdx.y < TILE_HEIGHT) {
        int apron_y = min(height - 1, global_y + radius);
        if (global_x < width && apron_y < height) {
            s_col[s_y + radius][threadIdx.x] = temp[coordsToIndex(global_x, apron_y, width)];
        }
    }
    
    __syncthreads();
    
    if (global_x < width && global_y < height) {
        float sum = 0.0f;
        
        #pragma unroll
        for (int k = -radius; k <= radius; k++) {
            sum += s_col[s_y + k][threadIdx.x];
        }
        
        int kernel_size = 2 * radius + 1;
        float mean = sum / (float)kernel_size;
        output[coordsToIndex(global_x, global_y, width)] = (unsigned char)clampf(mean, 0.0f, 255.0f);
    }
}

/**
 * Host function: Box blur
 */
extern "C" float boxBlurCUDA(
    const unsigned char* h_input,
    unsigned char* h_output,
    int width,
    int height,
    int radius,
    bool use_separable = true
) {
    if (radius > GAUSSIAN_KERNEL_RADIUS) {
        fprintf(stderr, "Warning: Radius too large, clamping to %d\n", GAUSSIAN_KERNEL_RADIUS);
        radius = GAUSSIAN_KERNEL_RADIUS;
    }
    
    size_t image_bytes = width * height * sizeof(unsigned char);
    
    unsigned char *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, image_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, image_bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, image_bytes, cudaMemcpyHostToDevice));
    
    CUDATimer timer;
    timer.start();
    
    if (use_separable) {
        // Separable implementation
        float *d_temp;
        size_t temp_bytes = width * height * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
        
        dim3 block_dim_h(TILE_WIDTH, 4);
        dim3 grid_dim_h((width + TILE_WIDTH - 1) / TILE_WIDTH, 
                        (height + block_dim_h.y - 1) / block_dim_h.y);
        
        dim3 block_dim_v(TILE_WIDTH, TILE_HEIGHT);
        dim3 grid_dim_v((width + TILE_WIDTH - 1) / TILE_WIDTH, 
                        (height + TILE_HEIGHT - 1) / TILE_HEIGHT);
        
        boxBlurHorizontal<<<grid_dim_h, block_dim_h>>>(d_input, d_temp, width, height, radius);
        CUDA_CHECK_LAST_ERROR();
        
        boxBlurVertical<<<grid_dim_v, block_dim_v>>>(d_temp, d_output, width, height, radius);
        CUDA_CHECK_LAST_ERROR();
        
        CUDA_CHECK(cudaFree(d_temp));
    } else {
        // 2D implementation
        dim3 block_dim(TILE_WIDTH, TILE_HEIGHT);
        dim3 grid_dim((width + TILE_WIDTH - 1) / TILE_WIDTH, 
                      (height + TILE_HEIGHT - 1) / TILE_HEIGHT);
        
        boxBlurKernel<<<grid_dim, block_dim>>>(d_input, d_output, width, height, radius);
        CUDA_CHECK_LAST_ERROR();
    }
    
    float elapsed_time = timer.stop();
    
    CUDA_CHECK(cudaMemcpy(h_output, d_output, image_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    return elapsed_time;
}
