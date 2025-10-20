#include "utils.cuh"

// Constant memory for 1D Gaussian kernel
__constant__ float d_gaussian_kernel[MAX_KERNEL_SIZE];

/**
 * Optimized horizontal Gaussian blur using shared memory
 * 
 * Uses 1D kernel and separable convolution for 2× speedup
 * Shared memory reduces global memory bandwidth
 */
__global__ void gaussianBlurHorizontal(
    const unsigned char* __restrict__ input,
    float* __restrict__ temp,
    int width,
    int height,
    int radius
) {
    // Shared memory with apron for radius
    __shared__ float s_row[TILE_WIDTH + 2 * GAUSSIAN_KERNEL_RADIUS];
    
    int global_x = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (global_y >= height) return;
    
    // Load center of tile
    int s_x = threadIdx.x + radius;
    if (global_x < width) {
        s_row[s_x] = (float)input[coordsToIndex(global_x, global_y, width)];
    } else {
        s_row[s_x] = 0.0f;
    }
    
    // Load left apron
    if (threadIdx.x < radius) {
        int apron_x = global_x - radius;
        if (apron_x >= 0) {
            s_row[threadIdx.x] = (float)input[coordsToIndex(apron_x, global_y, width)];
        } else {
            // Replicate edge
            s_row[threadIdx.x] = (float)input[coordsToIndex(0, global_y, width)];
        }
    }
    
    // Load right apron
    if (threadIdx.x >= TILE_WIDTH - radius && threadIdx.x < TILE_WIDTH) {
        int apron_x = global_x + radius;
        if (apron_x < width) {
            s_row[s_x + radius] = (float)input[coordsToIndex(apron_x, global_y, width)];
        } else {
            // Replicate edge
            s_row[s_x + radius] = (float)input[coordsToIndex(width - 1, global_y, width)];
        }
    }
    
    __syncthreads();
    
    // Compute horizontal blur
    if (global_x < width) {
        float sum = 0.0f;
        
        #pragma unroll
        for (int k = -radius; k <= radius; k++) {
            sum += s_row[s_x + k] * d_gaussian_kernel[k + radius];
        }
        
        temp[coordsToIndex(global_x, global_y, width)] = sum;
    }
}

/**
 * Optimized vertical Gaussian blur using shared memory
 * 
 * Second pass of separable convolution
 * Reads from temp buffer, writes final result
 */
__global__ void gaussianBlurVertical(
    const float* __restrict__ temp,
    unsigned char* __restrict__ output,
    int width,
    int height,
    int radius
) {
    // Shared memory for column
    __shared__ float s_col[TILE_HEIGHT + 2 * GAUSSIAN_KERNEL_RADIUS][TILE_WIDTH];
    
    int global_x = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int global_y = blockIdx.y * TILE_HEIGHT + threadIdx.y;
    
    int s_y = threadIdx.y + radius;
    
    // Load center of tile
    if (global_x < width && global_y < height) {
        s_col[s_y][threadIdx.x] = temp[coordsToIndex(global_x, global_y, width)];
    } else {
        s_col[s_y][threadIdx.x] = 0.0f;
    }
    
    // Load top apron
    if (threadIdx.y < radius) {
        int apron_y = global_y - radius;
        if (global_x < width && apron_y >= 0) {
            s_col[threadIdx.y][threadIdx.x] = temp[coordsToIndex(global_x, apron_y, width)];
        } else if (global_x < width) {
            // Replicate edge
            s_col[threadIdx.y][threadIdx.x] = temp[coordsToIndex(global_x, 0, width)];
        } else {
            s_col[threadIdx.y][threadIdx.x] = 0.0f;
        }
    }
    
    // Load bottom apron
    if (threadIdx.y >= TILE_HEIGHT - radius && threadIdx.y < TILE_HEIGHT) {
        int apron_y = global_y + radius;
        if (global_x < width && apron_y < height) {
            s_col[s_y + radius][threadIdx.x] = temp[coordsToIndex(global_x, apron_y, width)];
        } else if (global_x < width) {
            // Replicate edge
            s_col[s_y + radius][threadIdx.x] = temp[coordsToIndex(global_x, height - 1, width)];
        } else {
            s_col[s_y + radius][threadIdx.x] = 0.0f;
        }
    }
    
    __syncthreads();
    
    // Compute vertical blur
    if (global_x < width && global_y < height) {
        float sum = 0.0f;
        
        #pragma unroll
        for (int k = -radius; k <= radius; k++) {
            sum += s_col[s_y + k][threadIdx.x] * d_gaussian_kernel[k + radius];
        }
        
        output[coordsToIndex(global_x, global_y, width)] = (unsigned char)clampf(sum, 0.0f, 255.0f);
    }
}

/**
 * Fused Gaussian blur kernel (single pass, less memory efficient but simpler)
 * Uses 2D kernel - useful for small kernels or debugging
 */
__global__ void gaussianBlurFused(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int width,
    int height,
    const float* __restrict__ kernel_2d,
    int radius
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float sum = 0.0f;
    int kernel_size = 2 * radius + 1;
    
    for (int ky = -radius; ky <= radius; ky++) {
        for (int kx = -radius; kx <= radius; kx++) {
            int nx = clamp(x + kx, 0, width - 1);
            int ny = clamp(y + ky, 0, height - 1);
            
            int kernel_idx = (ky + radius) * kernel_size + (kx + radius);
            float pixel = (float)input[coordsToIndex(nx, ny, width)];
            sum += pixel * kernel_2d[kernel_idx];
        }
    }
    
    output[coordsToIndex(x, y, width)] = (unsigned char)clampf(sum, 0.0f, 255.0f);
}

/**
 * Host function: Gaussian blur with separable convolution
 * 
 * @param h_input: Host input image (grayscale)
 * @param h_output: Host output image
 * @param width: Image width
 * @param height: Image height
 * @param sigma: Gaussian standard deviation
 * @return: Execution time in milliseconds
 */
extern "C" float gaussianBlurCUDA(
    const unsigned char* h_input,
    unsigned char* h_output,
    int width,
    int height,
    float sigma
) {
    // Calculate kernel radius (3 sigma rule)
    int radius = (int)ceilf(3.0f * sigma);
    if (radius > GAUSSIAN_KERNEL_RADIUS) {
        fprintf(stderr, "Warning: Sigma too large, clamping radius to %d\n", GAUSSIAN_KERNEL_RADIUS);
        radius = GAUSSIAN_KERNEL_RADIUS;
    }
    
    int kernel_size = 2 * radius + 1;
    
    // Generate 1D Gaussian kernel
    float h_kernel[MAX_KERNEL_SIZE];
    generateGaussianKernel(h_kernel, radius, sigma);
    
    // Allocate device memory
    size_t image_bytes = width * height * sizeof(unsigned char);
    size_t temp_bytes = width * height * sizeof(float);
    
    unsigned char *d_input, *d_output;
    float *d_temp;
    
    CUDA_CHECK(cudaMalloc(&d_input, image_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, image_bytes));
    CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
    
    // Copy input and kernel to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, image_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(d_gaussian_kernel, h_kernel, kernel_size * sizeof(float)));
    
    // Configure kernel launches
    dim3 block_dim_h(TILE_WIDTH, 4);  // Horizontal: more threads per row
    dim3 grid_dim_h(
        (width + TILE_WIDTH - 1) / TILE_WIDTH,
        (height + block_dim_h.y - 1) / block_dim_h.y
    );
    
    dim3 block_dim_v(TILE_WIDTH, TILE_HEIGHT);  // Vertical: standard tile
    dim3 grid_dim_v(
        (width + TILE_WIDTH - 1) / TILE_WIDTH,
        (height + TILE_HEIGHT - 1) / TILE_HEIGHT
    );
    
    // Benchmark execution
    CUDATimer timer;
    timer.start();
    
    // Two-pass separable convolution
    gaussianBlurHorizontal<<<grid_dim_h, block_dim_h>>>(
        d_input, d_temp, width, height, radius
    );
    CUDA_CHECK_LAST_ERROR();
    
    gaussianBlurVertical<<<grid_dim_v, block_dim_v>>>(
        d_temp, d_output, width, height, radius
    );
    CUDA_CHECK_LAST_ERROR();
    
    float elapsed_time = timer.stop();
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_output, d_output, image_bytes, cudaMemcpyDeviceToHost));
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_temp));
    
    return elapsed_time;
}

/**
 * Host function: Gaussian blur with 2D kernel (less optimized, for comparison)
 */
extern "C" float gaussianBlurFusedCUDA(
    const unsigned char* h_input,
    unsigned char* h_output,
    int width,
    int height,
    float sigma
) {
    int radius = (int)ceilf(3.0f * sigma);
    if (radius > GAUSSIAN_KERNEL_RADIUS) {
        radius = GAUSSIAN_KERNEL_RADIUS;
    }
    
    int kernel_size = 2 * radius + 1;
    
    // Generate 2D Gaussian kernel
    float h_kernel_2d[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];
    float sum = 0.0f;
    float two_sigma_sq = 2.0f * sigma * sigma;
    
    for (int y = 0; y < kernel_size; y++) {
        for (int x = 0; x < kernel_size; x++) {
            int dx = x - radius;
            int dy = y - radius;
            float val = expf(-(dx * dx + dy * dy) / two_sigma_sq);
            h_kernel_2d[y * kernel_size + x] = val;
            sum += val;
        }
    }
    
    // Normalize
    for (int i = 0; i < kernel_size * kernel_size; i++) {
        h_kernel_2d[i] /= sum;
    }
    
    // Allocate device memory
    size_t image_bytes = width * height * sizeof(unsigned char);
    size_t kernel_bytes = kernel_size * kernel_size * sizeof(float);
    
    unsigned char *d_input, *d_output;
    float *d_kernel_2d;
    
    CUDA_CHECK(cudaMalloc(&d_input, image_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, image_bytes));
    CUDA_CHECK(cudaMalloc(&d_kernel_2d, kernel_bytes));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, image_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel_2d, h_kernel_2d, kernel_bytes, cudaMemcpyHostToDevice));
    
    dim3 block_dim(16, 16);
    dim3 grid_dim(
        (width + block_dim.x - 1) / block_dim.x,
        (height + block_dim.y - 1) / block_dim.y
    );
    
    CUDATimer timer;
    timer.start();
    
    gaussianBlurFused<<<grid_dim, block_dim>>>(
        d_input, d_output, width, height, d_kernel_2d, radius
    );
    CUDA_CHECK_LAST_ERROR();
    
    float elapsed_time = timer.stop();
    
    CUDA_CHECK(cudaMemcpy(h_output, d_output, image_bytes, cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_kernel_2d));
    
    return elapsed_time;
}
