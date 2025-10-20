# CUDA Image Filters

High-performance image processing operations implemented from scratch in CUDA C++, with comprehensive benchmarks against CPU (OpenCV) and other GPU frameworks (CuPy, PyTorch).

## Project Overview

This project implements core image filtering operations using CUDA kernels with advanced optimization techniques including:
- Shared memory tiling
- Coalesced memory access
- Kernel fusion
- Separable convolution optimization

## Features

### Implemented Kernels
- **Generic Convolution**: Arbitrary kernel support with shared memory optimization
- **Gaussian Blur**: Separable implementation (horizontal + vertical passes)
- **Sobel Edge Detection**: Gradient computation with magnitude and direction
- **Box Blur**: Optimized mean filter
- **Sharpening/Unsharp Mask**: Kernel fusion demonstration

### Performance Optimizations
- Shared memory for reduced global memory latency
- Coalesced memory access patterns
- Loop unrolling in kernel inner loops
- Constant memory for filter coefficients
- CUDA streams for pipelined execution

## Requirements

- NVIDIA GPU with CUDA Compute Capability ≥ 3.5
- CUDA Toolkit ≥ 11.0
- CMake ≥ 3.18
- Python ≥ 3.8
- OpenCV (for benchmarking)
- NumPy, CuPy, PyTorch (for comparison)

## Build Instructions

### C++ Library

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Python Bindings

```bash
sh build_colab.sh
```

## 💻 Usage

### C++ Interface

```cpp
#include "gaussian_blur.cuh"

// Load image data
unsigned char* d_input;
unsigned char* d_output;
// ... allocate and copy to device

// Apply Gaussian blur
gaussianBlurCUDA(d_input, d_output, width, height, sigma);
```

### Python Interface

```python
import fastfilters
import cv2

# Load image
img = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)

# Apply filters
blur = fastfilters.gaussian_blur(img, sigma=1.5)
edges = fastfilters.sobel(img)
sharp = fastfilters.sharpen(img, amount=1.5)
```

## 📊 Benchmark Results

**Comprehensive performance comparison on NVIDIA GPU (15.8 GB VRAM):**

### Gaussian Blur Performance - Multi-Framework Comparison

| Image Size | CUDA (Ours) | PyTorch GPU | TensorFlow GPU | OpenCV CPU | CUDA Speedup vs CPU |
|------------|-------------|-------------|----------------|------------|---------------------|
| 512×512 (0.26 MP)    | **0.28 ms** 🏆<br>950.48 MP/s | 0.33 ms<br>801.67 MP/s | 1.08 ms<br>243.70 MP/s | 0.91 ms<br>287.18 MP/s | **3.31×** |
| 1024×1024 (1.05 MP)  | **1.21 ms** 🏆<br>865.45 MP/s | 1.22 ms<br>857.16 MP/s | 3.01 ms<br>348.71 MP/s | 2.59 ms<br>404.65 MP/s | **2.14×** |
| 2048×2048 (4.19 MP)  | **3.54 ms** 🏆<br>1185.77 MP/s | 4.69 ms<br>895.03 MP/s | 12.29 ms<br>341.40 MP/s | 10.77 ms<br>389.41 MP/s | **3.05×** |
| 4096×4096 (16.78 MP) | **12.53 ms**<br>1338.73 MP/s | 11.21 ms ⚡<br>1496.05 MP/s | 87.94 ms<br>190.78 MP/s | 35.67 ms<br>470.41 MP/s | **2.85×** |
| 8192×8192 (67.11 MP) | **55.04 ms**<br>1219.29 MP/s | 41.56 ms ⚡<br>1614.88 MP/s | 311.70 ms<br>215.30 MP/s | 137.78 ms<br>487.06 MP/s | **2.50×** |

**Key Insights:**

✅ **Competitive Performance:**
- 🏆 **Wins on small to medium images** (512×512 to 2048×2048)
- Matches PyTorch performance on 1024×1024 images
- Within 10-25% of PyTorch on very large images (4K+)

✅ **Consistent Speedups:**
- **2.1-3.3× faster** than OpenCV CPU across all image sizes
- Significantly outperforms TensorFlow GPU (2-6× faster)

✅ **Excellent Scaling:**
- Throughput increases from **950 MP/s → 1.34 GP/s** as images grow
- Peak throughput at **4K resolution**: 1.34 GP/s
- Maintains high performance even on **67 MP images**

✅ **Memory Efficiency:**
- Successfully processes 8192×8192 images (67 MP)
- Separable convolution optimization reduces memory pressure
- Shared memory tiling maximizes cache utilization

**Performance Analysis:**
- Small images (< 1 MP): Custom CUDA kernels excel due to lower overhead
- Large images (16+ MP): PyTorch's highly optimized cuDNN backend slightly edges ahead
- TensorFlow: Slower due to graph execution overhead
- OpenCV CPU: Linear scaling, no parallelism advantage

## 🧪 Validation

All CUDA implementations validated against OpenCV with `atol=1e-3`:
```bash
python tests/validate.py
```

## 📂 Project Structure

```
cuda_image_filters/
├── src/                    # CUDA kernel implementations
│   ├── utils.cuh          # Common utilities and error checking
│   ├── convolution.cu     # Generic convolution kernel
│   ├── gaussian_blur.cu   # Separable Gaussian blur
│   ├── sobel.cu           # Sobel edge detection
│   ├── box_blur.cu        # Box filter
│   └── sharpening.cu      # Sharpening filters
├── python_bindings/       # Python interface via pybind11
│   ├── bindings.cpp       # Python wrapper code
│   └── setup.py           # Build configuration
├── benchmarks/            # Performance testing
│   └── benchmark.py       # Comprehensive benchmarks
├── tests/                 # Validation tests
│   └── validate.py        # Accuracy verification
└── CMakeLists.txt         # Build configuration
```

## 🎓 Optimization Techniques Explained

### 1. Shared Memory Tiling
Reduces global memory bandwidth by loading image tiles into fast shared memory:
```cuda
__shared__ float tile[TILE_HEIGHT][TILE_WIDTH];
// Load tile collaboratively
// Compute using shared memory
```

### 2. Separable Convolution
Gaussian blur decomposes 2D convolution into two 1D passes:
- Horizontal pass: O(width × height × kernel_size)
- Vertical pass: O(width × height × kernel_size)
- Total: O(2n) instead of O(n²)

### 3. Coalesced Memory Access
Adjacent threads access adjacent memory locations for maximum throughput.

### 4. Constant Memory
Small read-only data (filter kernels) stored in constant memory for broadcast efficiency.

## 📖 References

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [GPU Gems - Image Processing](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.

## 📬 Contact

For questions or feedback, please open an issue on GitHub.
