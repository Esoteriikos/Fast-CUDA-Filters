#include "utils.cuh"

// Forward declaration from gaussian_blur.cu
extern "C" float gaussianBlurCUDA(
    const unsigned char* h_input,
    unsigned char* h_output,
    int width,
    int height,
    float sigma
);

/**
 * Unsharp masking kernel - fused implementation
 * 
 * Formula: sharpened = original + amount * (original - blurred)
 * 
 * This kernel demonstrates kernel fusion:
 * Instead of: blur → subtract → scale → add (4 passes)
 * We do: blur → compute_sharpen (2 passes)
 */
__global__ void unsharpMaskKernel(
    const unsigned char* __restrict__ original,
    const float* __restrict__ blurred,
    unsigned char* __restrict__ output,
    int width,
    int height,
    float amount
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = coordsToIndex(x, y, width);
    
    float orig = (float)original[idx];
    float blur = blurred[idx];
    
    // Unsharp mask formula
    float sharpened = orig + amount * (orig - blur);
    
    output[idx] = (unsigned char)clampf(sharpened, 0.0f, 255.0f);
}

/**
 * Simple sharpening kernel using Laplacian
 * 
 * Laplacian kernel:
 * [ 0 -1  0]
 * [-1  4 -1]
 * [ 0 -1  0]
 * 
 * Result: original + amount * laplacian
 */
__global__ void laplacianSharpenKernel(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int width,
    int height,
    float amount
) {
    __shared__ float s_tile[TILE_HEIGHT + 2][TILE_WIDTH + 2];
    
    int global_x = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int global_y = blockIdx.y * TILE_HEIGHT + threadIdx.y;
    
    int s_x = threadIdx.x + 1;
    int s_y = threadIdx.y + 1;
    
    // Load tile center
    if (global_x < width && global_y < height) {
        s_tile[s_y][s_x] = (float)input[coordsToIndex(global_x, global_y, width)];
    } else {
        s_tile[s_y][s_x] = 0.0f;
    }
    
    // Load apron
    if (threadIdx.x == 0 && global_x > 0) {
        s_tile[s_y][0] = (float)input[coordsToIndex(global_x - 1, global_y, width)];
    } else if (threadIdx.x == 0) {
        s_tile[s_y][0] = s_tile[s_y][s_x];
    }
    
    if (threadIdx.x == TILE_WIDTH - 1 && global_x < width - 1) {
        s_tile[s_y][s_x + 1] = (float)input[coordsToIndex(global_x + 1, global_y, width)];
    } else if (threadIdx.x == TILE_WIDTH - 1) {
        s_tile[s_y][s_x + 1] = s_tile[s_y][s_x];
    }
    
    if (threadIdx.y == 0 && global_y > 0) {
        s_tile[0][s_x] = (float)input[coordsToIndex(global_x, global_y - 1, width)];
    } else if (threadIdx.y == 0) {
        s_tile[0][s_x] = s_tile[s_y][s_x];
    }
    
    if (threadIdx.y == TILE_HEIGHT - 1 && global_y < height - 1) {
        s_tile[s_y + 1][s_x] = (float)input[coordsToIndex(global_x, global_y + 1, width)];
    } else if (threadIdx.y == TILE_HEIGHT - 1) {
        s_tile[s_y + 1][s_x] = s_tile[s_y][s_x];
    }
    
    __syncthreads();
    
    if (global_x < width && global_y < height) {
        // Apply Laplacian
        float laplacian = -s_tile[s_y - 1][s_x]      // top
                          -s_tile[s_y][s_x - 1]      // left
                          +4.0f * s_tile[s_y][s_x]   // center
                          -s_tile[s_y][s_x + 1]      // right
                          -s_tile[s_y + 1][s_x];     // bottom
        
        float original = s_tile[s_y][s_x];
        float sharpened = original + amount * laplacian;
        
        output[coordsToIndex(global_x, global_y, width)] = (unsigned char)clampf(sharpened, 0.0f, 255.0f);
    }
}

/**
 * High-pass filter kernel (alternative sharpening approach)
 * 
 * Emphasizes high-frequency details
 */
__global__ void highPassKernel(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int width,
    int height
) {
    __shared__ float s_tile[TILE_HEIGHT + 2][TILE_WIDTH + 2];
    
    int global_x = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int global_y = blockIdx.y * TILE_HEIGHT + threadIdx.y;
    
    int s_x = threadIdx.x + 1;
    int s_y = threadIdx.y + 1;
    
    // Load tile
    if (global_x < width && global_y < height) {
        s_tile[s_y][s_x] = (float)input[coordsToIndex(global_x, global_y, width)];
    } else {
        s_tile[s_y][s_x] = 0.0f;
    }
    
    // Simplified apron loading (replicate edges)
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
        // High-pass kernel: [-1 -1 -1; -1 8 -1; -1 -1 -1]
        float result = -s_tile[s_y - 1][s_x - 1] - s_tile[s_y - 1][s_x] - s_tile[s_y - 1][s_x + 1]
                       -s_tile[s_y][s_x - 1]     + 8.0f * s_tile[s_y][s_x] - s_tile[s_y][s_x + 1]
                       -s_tile[s_y + 1][s_x - 1] - s_tile[s_y + 1][s_x] - s_tile[s_y + 1][s_x + 1];
        
        output[coordsToIndex(global_x, global_y, width)] = (unsigned char)clampf(result, 0.0f, 255.0f);
    }
}

/**
 * Host function: Unsharp masking
 * 
 * @param h_input: Input image
 * @param h_output: Sharpened output
 * @param width: Image width
 * @param height: Image height
 * @param sigma: Gaussian blur sigma for mask
 * @param amount: Sharpening strength (1.0-2.0 typical)
 * @return: Execution time
 */
extern "C" float sharpenUnsharpMaskCUDA(
    const unsigned char* h_input,
    unsigned char* h_output,
    int width,
    int height,
    float sigma,
    float amount
) {
    size_t image_bytes = width * height * sizeof(unsigned char);
    size_t float_bytes = width * height * sizeof(float);
    
    // Allocate device memory
    unsigned char *d_input, *d_output, *d_blurred;
    float *d_blurred_float;
    
    CUDA_CHECK(cudaMalloc(&d_input, image_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, image_bytes));
    CUDA_CHECK(cudaMalloc(&d_blurred, image_bytes));
    CUDA_CHECK(cudaMalloc(&d_blurred_float, float_bytes));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, image_bytes, cudaMemcpyHostToDevice));
    
    CUDATimer timer;
    timer.start();
    
    // Step 1: Blur image (using existing Gaussian blur)
    unsigned char* temp_blurred = new unsigned char[width * height];
    gaussianBlurCUDA(h_input, temp_blurred, width, height, sigma);
    CUDA_CHECK(cudaMemcpy(d_blurred, temp_blurred, image_bytes, cudaMemcpyHostToDevice));
    
    // Convert to float for computation
    dim3 block_dim(16, 16);
    dim3 grid_dim((width + 15) / 16, (height + 15) / 16);
    
    // Simple conversion kernel (inline lambda)
    auto convertToFloat = [&]() {
        auto kernel = [] __device__ (const unsigned char* in, float* out, int w, int h) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x < w && y < h) {
                int idx = y * w + x;
                out[idx] = (float)in[idx];
            }
        };
    };
    
    // Simple approach: convert on CPU
    float* temp_float = new float[width * height];
    for (int i = 0; i < width * height; i++) {
        temp_float[i] = (float)temp_blurred[i];
    }
    CUDA_CHECK(cudaMemcpy(d_blurred_float, temp_float, float_bytes, cudaMemcpyHostToDevice));
    
    // Step 2: Apply unsharp mask
    unsharpMaskKernel<<<grid_dim, block_dim>>>(
        d_input, d_blurred_float, d_output, width, height, amount
    );
    CUDA_CHECK_LAST_ERROR();
    
    float elapsed_time = timer.stop();
    
    CUDA_CHECK(cudaMemcpy(h_output, d_output, image_bytes, cudaMemcpyDeviceToHost));
    
    delete[] temp_blurred;
    delete[] temp_float;
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_blurred));
    CUDA_CHECK(cudaFree(d_blurred_float));
    
    return elapsed_time;
}

/**
 * Host function: Laplacian sharpening
 */
extern "C" float sharpenLaplacianCUDA(
    const unsigned char* h_input,
    unsigned char* h_output,
    int width,
    int height,
    float amount
) {
    size_t image_bytes = width * height * sizeof(unsigned char);
    
    unsigned char *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, image_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, image_bytes));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, image_bytes, cudaMemcpyHostToDevice));
    
    dim3 block_dim(TILE_WIDTH, TILE_HEIGHT);
    dim3 grid_dim((width + TILE_WIDTH - 1) / TILE_WIDTH, 
                  (height + TILE_HEIGHT - 1) / TILE_HEIGHT);
    
    CUDATimer timer;
    timer.start();
    
    laplacianSharpenKernel<<<grid_dim, block_dim>>>(
        d_input, d_output, width, height, amount
    );
    CUDA_CHECK_LAST_ERROR();
    
    float elapsed_time = timer.stop();
    
    CUDA_CHECK(cudaMemcpy(h_output, d_output, image_bytes, cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    return elapsed_time;
}
