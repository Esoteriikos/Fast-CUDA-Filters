#ifndef UTILS_CUH
#define UTILS_CUH

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Kernel launch error checking
#define CUDA_CHECK_LAST_ERROR() \
    do { \
        cudaError_t error = cudaGetLastError(); \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA kernel launch error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Common constants
#define TILE_WIDTH 16
#define TILE_HEIGHT 16
#define MAX_KERNEL_SIZE 15
#define GAUSSIAN_KERNEL_RADIUS 7  // For sigma up to ~2.0

// Device inline functions for clamping and boundary handling
__device__ __forceinline__ int clamp(int val, int min_val, int max_val) {
    return max(min_val, min(val, max_val));
}

__device__ __forceinline__ float clampf(float val, float min_val, float max_val) {
    return fmaxf(min_val, fminf(val, max_val));
}

// Convert 2D coordinates to 1D index
__device__ __forceinline__ int coordsToIndex(int x, int y, int width) {
    return y * width + x;
}

// Boundary handling modes
enum BoundaryMode {
    BORDER_CONSTANT = 0,  // Fill with constant value
    BORDER_REPLICATE = 1,  // Replicate edge pixels
    BORDER_REFLECT = 2,    // Reflect across edge
    BORDER_WRAP = 3        // Wrap around
};

// Get pixel with boundary handling
__device__ __forceinline__ unsigned char getPixel(
    const unsigned char* image, 
    int x, int y, 
    int width, int height,
    BoundaryMode mode = BORDER_REPLICATE,
    unsigned char constant_value = 0
) {
    if (x >= 0 && x < width && y >= 0 && y < height) {
        return image[coordsToIndex(x, y, width)];
    }
    
    switch (mode) {
        case BORDER_CONSTANT:
            return constant_value;
        
        case BORDER_REPLICATE:
            x = clamp(x, 0, width - 1);
            y = clamp(y, 0, height - 1);
            return image[coordsToIndex(x, y, width)];
        
        case BORDER_REFLECT:
            if (x < 0) x = -x - 1;
            if (x >= width) x = 2 * width - x - 1;
            if (y < 0) y = -y - 1;
            if (y >= height) y = 2 * height - y - 1;
            x = clamp(x, 0, width - 1);
            y = clamp(y, 0, height - 1);
            return image[coordsToIndex(x, y, width)];
        
        case BORDER_WRAP:
            x = (x + width) % width;
            y = (y + height) % height;
            return image[coordsToIndex(x, y, width)];
        
        default:
            return 0;
    }
}

// Float version for intermediate computations
__device__ __forceinline__ float getPixelFloat(
    const float* image, 
    int x, int y, 
    int width, int height,
    BoundaryMode mode = BORDER_REPLICATE
) {
    if (x >= 0 && x < width && y >= 0 && y < height) {
        return image[coordsToIndex(x, y, width)];
    }
    
    switch (mode) {
        case BORDER_CONSTANT:
            return 0.0f;
        
        case BORDER_REPLICATE:
            x = clamp(x, 0, width - 1);
            y = clamp(y, 0, height - 1);
            return image[coordsToIndex(x, y, width)];
        
        case BORDER_REFLECT:
            if (x < 0) x = -x - 1;
            if (x >= width) x = 2 * width - x - 1;
            if (y < 0) y = -y - 1;
            if (y >= height) y = 2 * height - y - 1;
            x = clamp(x, 0, width - 1);
            y = clamp(y, 0, height - 1);
            return image[coordsToIndex(x, y, width)];
        
        default:
            return 0.0f;
    }
}

// Timer utility for benchmarking
class CUDATimer {
private:
    cudaEvent_t start_event, stop_event;
    
public:
    CUDATimer() {
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));
    }
    
    ~CUDATimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }
    
    void start() {
        CUDA_CHECK(cudaEventRecord(start_event, 0));
    }
    
    float stop() {
        float elapsed_time;
        CUDA_CHECK(cudaEventRecord(stop_event, 0));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));
        return elapsed_time;
    }
};

// Memory allocation helpers
template<typename T>
T* allocateDeviceMemory(size_t num_elements) {
    T* d_ptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, num_elements * sizeof(T)));
    return d_ptr;
}

template<typename T>
void copyToDevice(T* d_dst, const T* h_src, size_t num_elements) {
    CUDA_CHECK(cudaMemcpy(d_dst, h_src, num_elements * sizeof(T), 
                          cudaMemcpyHostToDevice));
}

template<typename T>
void copyToHost(T* h_dst, const T* d_src, size_t num_elements) {
    CUDA_CHECK(cudaMemcpy(h_dst, d_src, num_elements * sizeof(T), 
                          cudaMemcpyDeviceToHost));
}

template<typename T>
void freeDeviceMemory(T* d_ptr) {
    if (d_ptr) {
        CUDA_CHECK(cudaFree(d_ptr));
    }
}

// Generate 1D Gaussian kernel
inline void generateGaussianKernel(float* kernel, int radius, float sigma) {
    int size = 2 * radius + 1;
    float sum = 0.0f;
    float two_sigma_sq = 2.0f * sigma * sigma;
    
    for (int i = 0; i < size; i++) {
        int x = i - radius;
        kernel[i] = expf(-(x * x) / two_sigma_sq);
        sum += kernel[i];
    }
    
    // Normalize
    for (int i = 0; i < size; i++) {
        kernel[i] /= sum;
    }
}

// Calculate optimal grid and block dimensions
inline dim3 calculateGridDim(int width, int height, dim3 block_dim) {
    return dim3(
        (width + block_dim.x - 1) / block_dim.x,
        (height + block_dim.y - 1) / block_dim.y
    );
}

// Print device properties
inline void printDeviceProperties() {
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    
    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        
        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
        printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Memory Clock Rate: %.2f GHz\n", prop.memoryClockRate / 1e6);
        printf("  Memory Bus Width: %d-bit\n", prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth: %.2f GB/s\n", 
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    }
}

#endif // UTILS_CUH
