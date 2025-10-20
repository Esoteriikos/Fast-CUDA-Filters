#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "../src/cuda_filters.h"

namespace py = pybind11;

// Wrapper for Gaussian Blur
py::array_t<unsigned char> py_gaussian_blur(
    py::array_t<unsigned char> input,
    float sigma
) {
    py::buffer_info buf = input.request();
    
    if (buf.ndim != 2) {
        throw std::runtime_error("Input must be 2D array (grayscale image)");
    }
    
    int height = buf.shape[0];
    int width = buf.shape[1];
    
    auto result = py::array_t<unsigned char>(buf.size);
    py::buffer_info out_buf = result.request();
    
    unsigned char* input_ptr = static_cast<unsigned char*>(buf.ptr);
    unsigned char* output_ptr = static_cast<unsigned char*>(out_buf.ptr);
    
    float time = gaussianBlurCUDA(input_ptr, output_ptr, width, height, sigma);
    
    result.resize({height, width});
    return result;
}

// Wrapper for Sobel Edge Detection
py::array_t<unsigned char> py_sobel(
    py::array_t<unsigned char> input,
    bool use_accurate = true
) {
    py::buffer_info buf = input.request();
    
    if (buf.ndim != 2) {
        throw std::runtime_error("Input must be 2D array (grayscale image)");
    }
    
    int height = buf.shape[0];
    int width = buf.shape[1];
    
    auto result = py::array_t<unsigned char>(buf.size);
    py::buffer_info out_buf = result.request();
    
    unsigned char* input_ptr = static_cast<unsigned char*>(buf.ptr);
    unsigned char* output_ptr = static_cast<unsigned char*>(out_buf.ptr);
    
    float time = sobelCUDA(input_ptr, output_ptr, width, height, use_accurate);
    
    result.resize({height, width});
    return result;
}

// Wrapper for Box Blur
py::array_t<unsigned char> py_box_blur(
    py::array_t<unsigned char> input,
    int radius,
    bool use_separable = true
) {
    py::buffer_info buf = input.request();
    
    if (buf.ndim != 2) {
        throw std::runtime_error("Input must be 2D array (grayscale image)");
    }
    
    int height = buf.shape[0];
    int width = buf.shape[1];
    
    auto result = py::array_t<unsigned char>(buf.size);
    py::buffer_info out_buf = result.request();
    
    unsigned char* input_ptr = static_cast<unsigned char*>(buf.ptr);
    unsigned char* output_ptr = static_cast<unsigned char*>(out_buf.ptr);
    
    float time = boxBlurCUDA(input_ptr, output_ptr, width, height, radius, use_separable);
    
    result.resize({height, width});
    return result;
}

// Wrapper for Laplacian Sharpening
py::array_t<unsigned char> py_sharpen(
    py::array_t<unsigned char> input,
    float amount = 1.0f
) {
    py::buffer_info buf = input.request();
    
    if (buf.ndim != 2) {
        throw std::runtime_error("Input must be 2D array (grayscale image)");
    }
    
    int height = buf.shape[0];
    int width = buf.shape[1];
    
    auto result = py::array_t<unsigned char>(buf.size);
    py::buffer_info out_buf = result.request();
    
    unsigned char* input_ptr = static_cast<unsigned char*>(buf.ptr);
    unsigned char* output_ptr = static_cast<unsigned char*>(out_buf.ptr);
    
    float time = sharpenLaplacianCUDA(input_ptr, output_ptr, width, height, amount);
    
    result.resize({height, width});
    return result;
}

// Wrapper for Custom Convolution
py::array_t<unsigned char> py_convolve2d(
    py::array_t<unsigned char> input,
    py::array_t<float> kernel
) {
    py::buffer_info img_buf = input.request();
    py::buffer_info ker_buf = kernel.request();
    
    if (img_buf.ndim != 2) {
        throw std::runtime_error("Input must be 2D array (grayscale image)");
    }
    
    if (ker_buf.ndim != 2) {
        throw std::runtime_error("Kernel must be 2D array");
    }
    
    if (ker_buf.shape[0] != ker_buf.shape[1]) {
        throw std::runtime_error("Kernel must be square");
    }
    
    if (ker_buf.shape[0] % 2 == 0) {
        throw std::runtime_error("Kernel size must be odd");
    }
    
    int height = img_buf.shape[0];
    int width = img_buf.shape[1];
    int kernel_size = ker_buf.shape[0];
    
    auto result = py::array_t<unsigned char>(img_buf.size);
    py::buffer_info out_buf = result.request();
    
    unsigned char* input_ptr = static_cast<unsigned char*>(img_buf.ptr);
    unsigned char* output_ptr = static_cast<unsigned char*>(out_buf.ptr);
    float* kernel_ptr = static_cast<float*>(ker_buf.ptr);
    
    float time = convolution2DCUDA(input_ptr, output_ptr, width, height, 
                                    kernel_ptr, kernel_size, false);
    
    result.resize({height, width});
    return result;
}

// Wrapper for Sobel Gradients (returns multiple arrays)
py::tuple py_sobel_gradients(py::array_t<unsigned char> input) {
    py::buffer_info buf = input.request();
    
    if (buf.ndim != 2) {
        throw std::runtime_error("Input must be 2D array (grayscale image)");
    }
    
    int height = buf.shape[0];
    int width = buf.shape[1];
    size_t size = width * height;
    
    auto gx = py::array_t<float>(size);
    auto gy = py::array_t<float>(size);
    auto magnitude = py::array_t<float>(size);
    auto direction = py::array_t<float>(size);
    
    unsigned char* input_ptr = static_cast<unsigned char*>(buf.ptr);
    float* gx_ptr = static_cast<float*>(gx.request().ptr);
    float* gy_ptr = static_cast<float*>(gy.request().ptr);
    float* mag_ptr = static_cast<float*>(magnitude.request().ptr);
    float* dir_ptr = static_cast<float*>(direction.request().ptr);
    
    float time = sobelGradientsCUDA(input_ptr, gx_ptr, gy_ptr, mag_ptr, dir_ptr, 
                                     width, height);
    
    gx.resize({height, width});
    gy.resize({height, width});
    magnitude.resize({height, width});
    direction.resize({height, width});
    
    return py::make_tuple(gx, gy, magnitude, direction);
}

PYBIND11_MODULE(fastfilters, m) {
    m.doc() = "Fast CUDA-accelerated image filters";
    
    m.def("gaussian_blur", &py_gaussian_blur, 
          "Apply Gaussian blur to grayscale image",
          py::arg("image"), 
          py::arg("sigma") = 1.5f);
    
    m.def("sobel", &py_sobel, 
          "Apply Sobel edge detection",
          py::arg("image"), 
          py::arg("use_accurate") = true);
    
    m.def("box_blur", &py_box_blur, 
          "Apply box blur (mean filter)",
          py::arg("image"), 
          py::arg("radius") = 3,
          py::arg("use_separable") = true);
    
    m.def("sharpen", &py_sharpen, 
          "Apply Laplacian sharpening",
          py::arg("image"), 
          py::arg("amount") = 1.0f);
    
    m.def("convolve2d", &py_convolve2d, 
          "Apply custom 2D convolution",
          py::arg("image"), 
          py::arg("kernel"));
    
    m.def("sobel_gradients", &py_sobel_gradients, 
          "Compute Sobel gradients (returns gx, gy, magnitude, direction)",
          py::arg("image"));
}
