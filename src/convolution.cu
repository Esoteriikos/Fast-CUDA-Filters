#include "utils.cuh"

// Constant memory for kernel coefficients (faster for small read-only data)
__constant__ float d_kernel[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];

/**
 * Optimized 2D convolution kernel using shared memory tiling
 * 
 * Strategy:
 * 1. Load input tile to shared memory (including apron for kernel radius)
 * 2. Synchronize threads
 * 3. Compute convolution using shared memory (much faster than global)
 * 4. Write result to global memory
 * 
 * Memory access pattern ensures coalesced reads/writes
 */
__global__ void convolution2DKernel(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int width,
    int height,
    int kernel_radius
) {
    // Shared memory tile with apron
    __shared__ float s_tile[TILE_HEIGHT + 2 * MAX_KERNEL_SIZE][TILE_WIDTH + 2 * MAX_KERNEL_SIZE];
    
    // Global coordinates
    int global_x = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int global_y = blockIdx.y * TILE_HEIGHT + threadIdx.y;
    
    // Shared memory coordinates (with apron offset)
    int s_x = threadIdx.x + kernel_radius;
    int s_y = threadIdx.y + kernel_radius;
    
    int kernel_size = 2 * kernel_radius + 1;
    int apron_width = TILE_WIDTH + 2 * kernel_radius;
    int apron_height = TILE_HEIGHT + 2 * kernel_radius;
    
    // Collaboratively load tile center
    if (global_x < width && global_y < height) {
        s_tile[s_y][s_x] = (float)input[coordsToIndex(global_x, global_y, width)];
    } else {
        s_tile[s_y][s_x] = 0.0f;
    }
    
    // Load left apron
    if (threadIdx.x < kernel_radius) {
        int apron_x = global_x - kernel_radius;
        if (apron_x >= 0 && global_y < height) {
            s_tile[s_y][threadIdx.x] = (float)input[coordsToIndex(apron_x, global_y, width)];
        } else {
            s_tile[s_y][threadIdx.x] = 0.0f;
        }
    }
    
    // Load right apron
    if (threadIdx.x >= TILE_WIDTH - kernel_radius) {
        int apron_x = global_x + kernel_radius;
        if (apron_x < width && global_y < height) {
            s_tile[s_y][s_x + kernel_radius] = (float)input[coordsToIndex(apron_x, global_y, width)];
        } else {
            s_tile[s_y][s_x + kernel_radius] = 0.0f;
        }
    }
    
    // Load top apron
    if (threadIdx.y < kernel_radius) {
        int apron_y = global_y - kernel_radius;
        if (global_x < width && apron_y >= 0) {
            s_tile[threadIdx.y][s_x] = (float)input[coordsToIndex(global_x, apron_y, width)];
        } else {
            s_tile[threadIdx.y][s_x] = 0.0f;
        }
    }
    
    // Load bottom apron
    if (threadIdx.y >= TILE_HEIGHT - kernel_radius) {
        int apron_y = global_y + kernel_radius;
        if (global_x < width && apron_y < height) {
            s_tile[s_y + kernel_radius][s_x] = (float)input[coordsToIndex(global_x, apron_y, width)];
        } else {
            s_tile[s_y + kernel_radius][s_x] = 0.0f;
        }
    }
    
    // Load corners (4 corners)
    if (threadIdx.x < kernel_radius && threadIdx.y < kernel_radius) {
        // Top-left
        int apron_x = global_x - kernel_radius;
        int apron_y = global_y - kernel_radius;
        if (apron_x >= 0 && apron_y >= 0) {
            s_tile[threadIdx.y][threadIdx.x] = (float)input[coordsToIndex(apron_x, apron_y, width)];
        } else {
            s_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }
    }
    
    if (threadIdx.x >= TILE_WIDTH - kernel_radius && threadIdx.y < kernel_radius) {
        // Top-right
        int apron_x = global_x + kernel_radius;
        int apron_y = global_y - kernel_radius;
        if (apron_x < width && apron_y >= 0) {
            s_tile[threadIdx.y][s_x + kernel_radius] = (float)input[coordsToIndex(apron_x, apron_y, width)];
        } else {
            s_tile[threadIdx.y][s_x + kernel_radius] = 0.0f;
        }
    }
    
    if (threadIdx.x < kernel_radius && threadIdx.y >= TILE_HEIGHT - kernel_radius) {
        // Bottom-left
        int apron_x = global_x - kernel_radius;
        int apron_y = global_y + kernel_radius;
        if (apron_x >= 0 && apron_y < height) {
            s_tile[s_y + kernel_radius][threadIdx.x] = (float)input[coordsToIndex(apron_x, apron_y, width)];
        } else {
            s_tile[s_y + kernel_radius][threadIdx.x] = 0.0f;
        }
    }
    
    if (threadIdx.x >= TILE_WIDTH - kernel_radius && threadIdx.y >= TILE_HEIGHT - kernel_radius) {
        // Bottom-right
        int apron_x = global_x + kernel_radius;
        int apron_y = global_y + kernel_radius;
        if (apron_x < width && apron_y < height) {
            s_tile[s_y + kernel_radius][s_x + kernel_radius] = (float)input[coordsToIndex(apron_x, apron_y, width)];
        } else {
            s_tile[s_y + kernel_radius][s_x + kernel_radius] = 0.0f;
        }
    }
    
    // Synchronize to ensure all data is loaded
    __syncthreads();
    
    // Perform convolution if within bounds
    if (global_x < width && global_y < height) {
        float sum = 0.0f;
        
        // Unroll small kernels for better performance
        #pragma unroll
        for (int ky = -kernel_radius; ky <= kernel_radius; ky++) {
            #pragma unroll
            for (int kx = -kernel_radius; kx <= kernel_radius; kx++) {
                int kernel_idx = (ky + kernel_radius) * kernel_size + (kx + kernel_radius);
                float pixel = s_tile[s_y + ky][s_x + kx];
                sum += pixel * d_kernel[kernel_idx];
            }
        }
        
        // Clamp and write result
        output[coordsToIndex(global_x, global_y, width)] = (unsigned char)clampf(sum, 0.0f, 255.0f);
    }
}

/**
 * Naive convolution kernel (for comparison/debugging)
 * Direct global memory access without optimization
 */
__global__ void convolution2DNaive(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int width,
    int height,
    int kernel_radius
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int kernel_size = 2 * kernel_radius + 1;
    float sum = 0.0f;
    
    for (int ky = -kernel_radius; ky <= kernel_radius; ky++) {
        for (int kx = -kernel_radius; kx <= kernel_radius; kx++) {
            int nx = x + kx;
            int ny = y + ky;
            
            // Boundary handling: replicate edge
            nx = clamp(nx, 0, width - 1);
            ny = clamp(ny, 0, height - 1);
            
            int kernel_idx = (ky + kernel_radius) * kernel_size + (kx + kernel_radius);
            unsigned char pixel = input[coordsToIndex(nx, ny, width)];
            sum += (float)pixel * d_kernel[kernel_idx];
        }
    }
    
    output[coordsToIndex(x, y, width)] = (unsigned char)clampf(sum, 0.0f, 255.0f);
}

/**
 * Host function: Generic 2D convolution
 * 
 * @param h_input: Host input image (grayscale)
 * @param h_output: Host output image
 * @param width: Image width
 * @param height: Image height
 * @param h_kernel: Convolution kernel (row-major)
 * @param kernel_size: Kernel dimension (must be odd)
 * @param use_naive: Use naive implementation (for testing)
 * @return: Execution time in milliseconds
 */
extern "C" float convolution2DCUDA(
    const unsigned char* h_input,
    unsigned char* h_output,
    int width,
    int height,
    const float* h_kernel,
    int kernel_size,
    bool use_naive = false
) {
    // Validate inputs
    if (kernel_size % 2 == 0 || kernel_size > MAX_KERNEL_SIZE) {
        fprintf(stderr, "Error: Kernel size must be odd and <= %d\n", MAX_KERNEL_SIZE);
        return -1.0f;
    }
    
    int kernel_radius = kernel_size / 2;
    size_t image_bytes = width * height * sizeof(unsigned char);
    size_t kernel_bytes = kernel_size * kernel_size * sizeof(float);
    
    // Allocate device memory
    unsigned char *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, image_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, image_bytes));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, image_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(d_kernel, h_kernel, kernel_bytes));
    
    // Configure kernel launch
    dim3 block_dim(TILE_WIDTH, TILE_HEIGHT);
    dim3 grid_dim(
        (width + TILE_WIDTH - 1) / TILE_WIDTH,
        (height + TILE_HEIGHT - 1) / TILE_HEIGHT
    );
    
    // Benchmark execution
    CUDATimer timer;
    timer.start();
    
    if (use_naive) {
        convolution2DNaive<<<grid_dim, block_dim>>>(
            d_input, d_output, width, height, kernel_radius
        );
    } else {
        convolution2DKernel<<<grid_dim, block_dim>>>(
            d_input, d_output, width, height, kernel_radius
        );
    }
    
    CUDA_CHECK_LAST_ERROR();
    float elapsed_time = timer.stop();
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_output, d_output, image_bytes, cudaMemcpyDeviceToHost));
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    return elapsed_time;
}
