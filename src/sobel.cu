#include "utils.cuh"

/**
 * Sobel edge detection with fused gradient computation
 * 
 * Computes Gx and Gy gradients, then calculates magnitude and direction
 * Uses shared memory for efficient neighborhood access
 * 
 * Sobel kernels:
 * Gx = [-1  0  1]        Gy = [-1 -2 -1]
 *      [-2  0  2]             [ 0  0  0]
 *      [-1  0  1]             [ 1  2  1]
 */
__global__ void sobelEdgeDetection(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int width,
    int height,
    bool compute_magnitude = true
) {
    // Shared memory with 1-pixel apron
    __shared__ float s_tile[TILE_HEIGHT + 2][TILE_WIDTH + 2];
    
    int global_x = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int global_y = blockIdx.y * TILE_HEIGHT + threadIdx.y;
    
    int s_x = threadIdx.x + 1;
    int s_y = threadIdx.y + 1;
    
    // Load center tile
    if (global_x < width && global_y < height) {
        s_tile[s_y][s_x] = (float)input[coordsToIndex(global_x, global_y, width)];
    } else {
        s_tile[s_y][s_x] = 0.0f;
    }
    
    // Load apron borders
    if (threadIdx.x == 0) {
        // Left edge
        int apron_x = max(0, global_x - 1);
        if (global_y < height) {
            s_tile[s_y][0] = (float)input[coordsToIndex(apron_x, global_y, width)];
        } else {
            s_tile[s_y][0] = 0.0f;
        }
    }
    
    if (threadIdx.x == TILE_WIDTH - 1 || global_x == width - 1) {
        // Right edge
        int apron_x = min(width - 1, global_x + 1);
        if (global_y < height && apron_x < width) {
            s_tile[s_y][s_x + 1] = (float)input[coordsToIndex(apron_x, global_y, width)];
        } else {
            s_tile[s_y][s_x + 1] = 0.0f;
        }
    }
    
    if (threadIdx.y == 0) {
        // Top edge
        int apron_y = max(0, global_y - 1);
        if (global_x < width) {
            s_tile[0][s_x] = (float)input[coordsToIndex(global_x, apron_y, width)];
        } else {
            s_tile[0][s_x] = 0.0f;
        }
    }
    
    if (threadIdx.y == TILE_HEIGHT - 1 || global_y == height - 1) {
        // Bottom edge
        int apron_y = min(height - 1, global_y + 1);
        if (global_x < width && apron_y < height) {
            s_tile[s_y + 1][s_x] = (float)input[coordsToIndex(global_x, apron_y, width)];
        } else {
            s_tile[s_y + 1][s_x] = 0.0f;
        }
    }
    
    // Load corners
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        int apron_x = max(0, global_x - 1);
        int apron_y = max(0, global_y - 1);
        s_tile[0][0] = (float)input[coordsToIndex(apron_x, apron_y, width)];
    }
    
    if ((threadIdx.x == TILE_WIDTH - 1 || global_x == width - 1) && threadIdx.y == 0) {
        int apron_x = min(width - 1, global_x + 1);
        int apron_y = max(0, global_y - 1);
        if (apron_x < width) {
            s_tile[0][s_x + 1] = (float)input[coordsToIndex(apron_x, apron_y, width)];
        }
    }
    
    if (threadIdx.x == 0 && (threadIdx.y == TILE_HEIGHT - 1 || global_y == height - 1)) {
        int apron_x = max(0, global_x - 1);
        int apron_y = min(height - 1, global_y + 1);
        if (apron_y < height) {
            s_tile[s_y + 1][0] = (float)input[coordsToIndex(apron_x, apron_y, width)];
        }
    }
    
    if ((threadIdx.x == TILE_WIDTH - 1 || global_x == width - 1) && 
        (threadIdx.y == TILE_HEIGHT - 1 || global_y == height - 1)) {
        int apron_x = min(width - 1, global_x + 1);
        int apron_y = min(height - 1, global_y + 1);
        if (apron_x < width && apron_y < height) {
            s_tile[s_y + 1][s_x + 1] = (float)input[coordsToIndex(apron_x, apron_y, width)];
        }
    }
    
    __syncthreads();
    
    // Compute Sobel gradients
    if (global_x < width && global_y < height) {
        // Sobel Gx (horizontal gradient)
        float gx = -1.0f * s_tile[s_y - 1][s_x - 1] + 1.0f * s_tile[s_y - 1][s_x + 1]
                   -2.0f * s_tile[s_y][s_x - 1]     + 2.0f * s_tile[s_y][s_x + 1]
                   -1.0f * s_tile[s_y + 1][s_x - 1] + 1.0f * s_tile[s_y + 1][s_x + 1];
        
        // Sobel Gy (vertical gradient)
        float gy = -1.0f * s_tile[s_y - 1][s_x - 1] - 2.0f * s_tile[s_y - 1][s_x] - 1.0f * s_tile[s_y - 1][s_x + 1]
                   +1.0f * s_tile[s_y + 1][s_x - 1] + 2.0f * s_tile[s_y + 1][s_x] + 1.0f * s_tile[s_y + 1][s_x + 1];
        
        float magnitude;
        if (compute_magnitude) {
            // Compute magnitude: sqrt(Gx^2 + Gy^2)
            magnitude = sqrtf(gx * gx + gy * gy);
        } else {
            // Approximation: |Gx| + |Gy| (faster)
            magnitude = fabsf(gx) + fabsf(gy);
        }
        
        output[coordsToIndex(global_x, global_y, width)] = (unsigned char)clampf(magnitude, 0.0f, 255.0f);
    }
}

/**
 * Sobel with separate gradient outputs
 * Returns Gx, Gy, magnitude, and direction in separate buffers
 */
__global__ void sobelGradients(
    const unsigned char* __restrict__ input,
    float* __restrict__ gx_out,
    float* __restrict__ gy_out,
    float* __restrict__ magnitude_out,
    float* __restrict__ direction_out,
    int width,
    int height
) {
    __shared__ float s_tile[TILE_HEIGHT + 2][TILE_WIDTH + 2];
    
    int global_x = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int global_y = blockIdx.y * TILE_HEIGHT + threadIdx.y;
    
    int s_x = threadIdx.x + 1;
    int s_y = threadIdx.y + 1;
    
    // Load tile (same as above)
    if (global_x < width && global_y < height) {
        s_tile[s_y][s_x] = (float)input[coordsToIndex(global_x, global_y, width)];
    } else {
        s_tile[s_y][s_x] = 0.0f;
    }
    
    // Simplified apron loading for brevity (use replicate mode)
    if (threadIdx.x == 0) {
        int apron_x = max(0, global_x - 1);
        s_tile[s_y][0] = global_y < height ? (float)input[coordsToIndex(apron_x, global_y, width)] : 0.0f;
    }
    if (threadIdx.x == TILE_WIDTH - 1 || global_x == width - 1) {
        int apron_x = min(width - 1, global_x + 1);
        s_tile[s_y][s_x + 1] = global_y < height && apron_x < width ? 
                                (float)input[coordsToIndex(apron_x, global_y, width)] : 0.0f;
    }
    if (threadIdx.y == 0) {
        int apron_y = max(0, global_y - 1);
        s_tile[0][s_x] = global_x < width ? (float)input[coordsToIndex(global_x, apron_y, width)] : 0.0f;
    }
    if (threadIdx.y == TILE_HEIGHT - 1 || global_y == height - 1) {
        int apron_y = min(height - 1, global_y + 1);
        s_tile[s_y + 1][s_x] = global_x < width && apron_y < height ? 
                                (float)input[coordsToIndex(global_x, apron_y, width)] : 0.0f;
    }
    
    __syncthreads();
    
    if (global_x < width && global_y < height) {
        float gx = -s_tile[s_y - 1][s_x - 1] + s_tile[s_y - 1][s_x + 1]
                   -2.0f * s_tile[s_y][s_x - 1] + 2.0f * s_tile[s_y][s_x + 1]
                   -s_tile[s_y + 1][s_x - 1] + s_tile[s_y + 1][s_x + 1];
        
        float gy = -s_tile[s_y - 1][s_x - 1] - 2.0f * s_tile[s_y - 1][s_x] - s_tile[s_y - 1][s_x + 1]
                   +s_tile[s_y + 1][s_x - 1] + 2.0f * s_tile[s_y + 1][s_x] + s_tile[s_y + 1][s_x + 1];
        
        int idx = coordsToIndex(global_x, global_y, width);
        
        gx_out[idx] = gx;
        gy_out[idx] = gy;
        magnitude_out[idx] = sqrtf(gx * gx + gy * gy);
        direction_out[idx] = atan2f(gy, gx);  // Radians
    }
}

/**
 * Host function: Sobel edge detection
 * 
 * @param h_input: Host input image (grayscale)
 * @param h_output: Host output image (edge magnitude)
 * @param width: Image width
 * @param height: Image height
 * @param use_accurate: Use sqrt for magnitude (true) or approximation (false)
 * @return: Execution time in milliseconds
 */
extern "C" float sobelCUDA(
    const unsigned char* h_input,
    unsigned char* h_output,
    int width,
    int height,
    bool use_accurate = true
) {
    size_t image_bytes = width * height * sizeof(unsigned char);
    
    unsigned char *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, image_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, image_bytes));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, image_bytes, cudaMemcpyHostToDevice));
    
    dim3 block_dim(TILE_WIDTH, TILE_HEIGHT);
    dim3 grid_dim(
        (width + TILE_WIDTH - 1) / TILE_WIDTH,
        (height + TILE_HEIGHT - 1) / TILE_HEIGHT
    );
    
    CUDATimer timer;
    timer.start();
    
    sobelEdgeDetection<<<grid_dim, block_dim>>>(
        d_input, d_output, width, height, use_accurate
    );
    CUDA_CHECK_LAST_ERROR();
    
    float elapsed_time = timer.stop();
    
    CUDA_CHECK(cudaMemcpy(h_output, d_output, image_bytes, cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    return elapsed_time;
}

/**
 * Host function: Sobel with full gradient information
 */
extern "C" float sobelGradientsCUDA(
    const unsigned char* h_input,
    float* h_gx,
    float* h_gy,
    float* h_magnitude,
    float* h_direction,
    int width,
    int height
) {
    size_t image_bytes = width * height * sizeof(unsigned char);
    size_t float_bytes = width * height * sizeof(float);
    
    unsigned char *d_input;
    float *d_gx, *d_gy, *d_mag, *d_dir;
    
    CUDA_CHECK(cudaMalloc(&d_input, image_bytes));
    CUDA_CHECK(cudaMalloc(&d_gx, float_bytes));
    CUDA_CHECK(cudaMalloc(&d_gy, float_bytes));
    CUDA_CHECK(cudaMalloc(&d_mag, float_bytes));
    CUDA_CHECK(cudaMalloc(&d_dir, float_bytes));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, image_bytes, cudaMemcpyHostToDevice));
    
    dim3 block_dim(TILE_WIDTH, TILE_HEIGHT);
    dim3 grid_dim(
        (width + TILE_WIDTH - 1) / TILE_WIDTH,
        (height + TILE_HEIGHT - 1) / TILE_HEIGHT
    );
    
    CUDATimer timer;
    timer.start();
    
    sobelGradients<<<grid_dim, block_dim>>>(
        d_input, d_gx, d_gy, d_mag, d_dir, width, height
    );
    CUDA_CHECK_LAST_ERROR();
    
    float elapsed_time = timer.stop();
    
    if (h_gx) CUDA_CHECK(cudaMemcpy(h_gx, d_gx, float_bytes, cudaMemcpyDeviceToHost));
    if (h_gy) CUDA_CHECK(cudaMemcpy(h_gy, d_gy, float_bytes, cudaMemcpyDeviceToHost));
    if (h_magnitude) CUDA_CHECK(cudaMemcpy(h_magnitude, d_mag, float_bytes, cudaMemcpyDeviceToHost));
    if (h_direction) CUDA_CHECK(cudaMemcpy(h_direction, d_dir, float_bytes, cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_gx));
    CUDA_CHECK(cudaFree(d_gy));
    CUDA_CHECK(cudaFree(d_mag));
    CUDA_CHECK(cudaFree(d_dir));
    
    return elapsed_time;
}
