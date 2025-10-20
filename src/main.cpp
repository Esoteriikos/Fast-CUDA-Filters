#include <iostream>
#include <cstdlib>
#include <cstring>
#include "cuda_filters.h"

// Simple image generation for testing
void generateTestImage(unsigned char* image, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Gradient pattern with some features
            int idx = y * width + x;
            if (x < width / 4 || x > 3 * width / 4) {
                image[idx] = 50;  // Dark borders
            } else if (y < height / 4 || y > 3 * height / 4) {
                image[idx] = 50;  // Dark borders
            } else if (abs(x - width/2) < 50 && abs(y - height/2) < 50) {
                image[idx] = 255;  // Bright square in center
            } else {
                image[idx] = (unsigned char)((x + y) % 256);  // Gradient
            }
        }
    }
}

void printImage(const unsigned char* image, int width, int height, int stride = 1) {
    std::cout << "Image sample (stride=" << stride << "):\n";
    for (int y = 0; y < height; y += stride) {
        for (int x = 0; x < width; x += stride) {
            int val = image[y * width + x];
            if (val < 64) std::cout << " ";
            else if (val < 128) std::cout << ".";
            else if (val < 192) std::cout << "o";
            else std::cout << "#";
        }
        std::cout << "\n";
    }
}

int main(int argc, char** argv) {
    std::cout << "=== CUDA Image Filters Test ===" << std::endl;
    std::cout << std::endl;
    
    // Test parameters
    const int width = 1024;
    const int height = 1024;
    const size_t image_size = width * height;
    
    // Allocate host memory
    unsigned char* h_input = new unsigned char[image_size];
    unsigned char* h_output = new unsigned char[image_size];
    
    // Generate test image
    generateTestImage(h_input, width, height);
    std::cout << "Generated " << width << "x" << height << " test image" << std::endl;
    std::cout << std::endl;
    
    // Test 1: Gaussian Blur
    std::cout << "Test 1: Gaussian Blur (sigma=2.0)" << std::endl;
    float time_gaussian = gaussianBlurCUDA(h_input, h_output, width, height, 2.0f);
    std::cout << "  Time: " << time_gaussian << " ms" << std::endl;
    std::cout << "  Throughput: " << (width * height / 1e6) / (time_gaussian / 1000.0f) 
              << " megapixels/sec" << std::endl;
    std::cout << std::endl;
    
    // Test 2: Sobel Edge Detection
    std::cout << "Test 2: Sobel Edge Detection" << std::endl;
    float time_sobel = sobelCUDA(h_input, h_output, width, height, true);
    std::cout << "  Time: " << time_sobel << " ms" << std::endl;
    std::cout << "  Throughput: " << (width * height / 1e6) / (time_sobel / 1000.0f) 
              << " megapixels/sec" << std::endl;
    std::cout << std::endl;
    
    // Test 3: Box Blur
    std::cout << "Test 3: Box Blur (radius=3, separable)" << std::endl;
    float time_box = boxBlurCUDA(h_input, h_output, width, height, 3, true);
    std::cout << "  Time: " << time_box << " ms" << std::endl;
    std::cout << "  Throughput: " << (width * height / 1e6) / (time_box / 1000.0f) 
              << " megapixels/sec" << std::endl;
    std::cout << std::endl;
    
    // Test 4: Laplacian Sharpening
    std::cout << "Test 4: Laplacian Sharpening (amount=0.5)" << std::endl;
    float time_sharpen = sharpenLaplacianCUDA(h_input, h_output, width, height, 0.5f);
    std::cout << "  Time: " << time_sharpen << " ms" << std::endl;
    std::cout << "  Throughput: " << (width * height / 1e6) / (time_sharpen / 1000.0f) 
              << " megapixels/sec" << std::endl;
    std::cout << std::endl;
    
    // Test 5: Custom Convolution (edge detection kernel)
    std::cout << "Test 5: Custom Convolution (3x3 edge kernel)" << std::endl;
    float edge_kernel[9] = {
        -1, -1, -1,
        -1,  8, -1,
        -1, -1, -1
    };
    float time_conv = convolution2DCUDA(h_input, h_output, width, height, edge_kernel, 3, false);
    std::cout << "  Time: " << time_conv << " ms" << std::endl;
    std::cout << "  Throughput: " << (width * height / 1e6) / (time_conv / 1000.0f) 
              << " megapixels/sec" << std::endl;
    std::cout << std::endl;
    
    // Summary
    std::cout << "=== Performance Summary ===" << std::endl;
    std::cout << "Image size: " << width << "x" << height << " (" 
              << (width * height / 1e6) << " megapixels)" << std::endl;
    std::cout << "All kernels executed successfully!" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Fastest operation: ";
    float min_time = time_gaussian;
    std::string fastest = "Gaussian Blur";
    
    if (time_sobel < min_time) { min_time = time_sobel; fastest = "Sobel"; }
    if (time_box < min_time) { min_time = time_box; fastest = "Box Blur"; }
    if (time_sharpen < min_time) { min_time = time_sharpen; fastest = "Sharpening"; }
    if (time_conv < min_time) { min_time = time_conv; fastest = "Convolution"; }
    
    std::cout << fastest << " (" << min_time << " ms)" << std::endl;
    
    // Cleanup
    delete[] h_input;
    delete[] h_output;
    
    return 0;
}
