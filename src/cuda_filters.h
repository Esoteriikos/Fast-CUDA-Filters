#ifndef CUDA_FILTERS_H
#define CUDA_FILTERS_H

#ifdef __cplusplus
extern "C" {
#endif

// Convolution
float convolution2DCUDA(
    const unsigned char* h_input,
    unsigned char* h_output,
    int width,
    int height,
    const float* h_kernel,
    int kernel_size,
    bool use_naive
);

// Gaussian Blur
float gaussianBlurCUDA(
    const unsigned char* h_input,
    unsigned char* h_output,
    int width,
    int height,
    float sigma
);

float gaussianBlurFusedCUDA(
    const unsigned char* h_input,
    unsigned char* h_output,
    int width,
    int height,
    float sigma
);

// Sobel Edge Detection
float sobelCUDA(
    const unsigned char* h_input,
    unsigned char* h_output,
    int width,
    int height,
    bool use_accurate
);

float sobelGradientsCUDA(
    const unsigned char* h_input,
    float* h_gx,
    float* h_gy,
    float* h_magnitude,
    float* h_direction,
    int width,
    int height
);

// Box Blur
float boxBlurCUDA(
    const unsigned char* h_input,
    unsigned char* h_output,
    int width,
    int height,
    int radius,
    bool use_separable
);

// Sharpening
float sharpenUnsharpMaskCUDA(
    const unsigned char* h_input,
    unsigned char* h_output,
    int width,
    int height,
    float sigma,
    float amount
);

float sharpenLaplacianCUDA(
    const unsigned char* h_input,
    unsigned char* h_output,
    int width,
    int height,
    float amount
);

#ifdef __cplusplus
}
#endif

#endif // CUDA_FILTERS_H
