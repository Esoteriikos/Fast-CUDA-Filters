import numpy as np
import time
import sys
from typing import Dict, List, Tuple

# Import frameworks
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    print("Warning: OpenCV not found")

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    print("Warning: CuPy not found")

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = torch.cuda.is_available()
    if not HAS_TORCH:
        print("Warning: PyTorch CUDA not available")
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not found")

try:
    import fastfilters
    HAS_CUDA_FILTERS = True
except ImportError:
    HAS_CUDA_FILTERS = False
    print("Warning: fastfilters not found. Build Python bindings first.")


class Benchmark:
    def __init__(self, image_sizes: List[Tuple[int, int]], warmup_runs: int = 3, benchmark_runs: int = 10):
        self.image_sizes = image_sizes
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.results = {}
        
    def generate_test_image(self, width: int, height: int) -> np.ndarray:
        """Generate synthetic test image with various features"""
        img = np.zeros((height, width), dtype=np.uint8)
        
        # Gradient
        for y in range(height):
            for x in range(width):
                img[y, x] = (x + y) % 256
        
        # Add some edges
        img[height//4:3*height//4, width//4] = 255
        img[height//4:3*height//4, 3*width//4] = 255
        img[height//4, width//4:3*width//4] = 255
        img[3*height//4, width//4:3*width//4] = 255
        
        # Add noise
        noise = np.random.randint(0, 50, (height, width), dtype=np.uint8)
        img = cv2.add(img, noise)
        
        return img
    
    def benchmark_function(self, func, *args, **kwargs) -> Tuple[float, np.ndarray]:
        """Benchmark a function with warmup and multiple runs"""
        # Warmup
        for _ in range(self.warmup_runs):
            result = func(*args, **kwargs)
        
        # Benchmark
        times = []
        for _ in range(self.benchmark_runs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            
            # Ensure GPU operations complete
            if HAS_CUPY and isinstance(result, cp.ndarray):
                cp.cuda.Stream.null.synchronize()
            elif HAS_TORCH and isinstance(result, torch.Tensor):
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        # Convert result to numpy if needed
        if HAS_CUPY and isinstance(result, cp.ndarray):
            result = cp.asnumpy(result)
        elif HAS_TORCH and isinstance(result, torch.Tensor):
            result = result.cpu().numpy()
        
        return avg_time, std_time, result
    
    def benchmark_gaussian_blur(self, img: np.ndarray, sigma: float = 2.0) -> Dict:
        """Benchmark Gaussian blur implementations"""
        results = {}
        height, width = img.shape
        
        # OpenCV CPU
        if HAS_OPENCV:
            ksize = int(6 * sigma + 1)
            if ksize % 2 == 0:
                ksize += 1
            
            avg_time, std_time, result = self.benchmark_function(
                cv2.GaussianBlur, img, (ksize, ksize), sigma
            )
            results['opencv_cpu'] = {'time': avg_time, 'std': std_time, 'result': result}
        
        # CuPy GPU
        if HAS_CUPY:
            img_gpu = cp.asarray(img)
            radius = int(3 * sigma)
            
            # Separable Gaussian using CuPy
            def cupy_gaussian(img_gpu, sigma):
                from cupyx.scipy.ndimage import gaussian_filter
                return gaussian_filter(img_gpu, sigma=sigma)
            
            avg_time, std_time, result = self.benchmark_function(cupy_gaussian, img_gpu, sigma)
            results['cupy_gpu'] = {'time': avg_time, 'std': std_time, 'result': result}
        
        # PyTorch GPU
        if HAS_TORCH:
            img_tensor = torch.from_numpy(img).float().cuda().unsqueeze(0).unsqueeze(0)
            ksize = int(6 * sigma + 1)
            if ksize % 2 == 0:
                ksize += 1
            
            # Create Gaussian kernel
            x = torch.arange(ksize, dtype=torch.float32, device='cuda') - ksize // 2
            gauss_1d = torch.exp(-(x ** 2) / (2 * sigma ** 2))
            gauss_1d = gauss_1d / gauss_1d.sum()
            
            def pytorch_gaussian(img_tensor, kernel):
                kernel_2d = kernel.unsqueeze(0) * kernel.unsqueeze(1)
                kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)
                padded = F.pad(img_tensor, (ksize//2, ksize//2, ksize//2, ksize//2), mode='replicate')
                result = F.conv2d(padded, kernel_2d)
                return result.squeeze().byte()
            
            avg_time, std_time, result = self.benchmark_function(pytorch_gaussian, img_tensor, gauss_1d)
            results['pytorch_gpu'] = {'time': avg_time, 'std': std_time, 'result': result}
        
        # CUDA (ours)
        if HAS_CUDA_FILTERS:
            avg_time, std_time, result = self.benchmark_function(
                fastfilters.gaussian_blur, img, sigma
            )
            results['cuda_ours'] = {'time': avg_time, 'std': std_time, 'result': result}
        
        return results
    
    def benchmark_sobel(self, img: np.ndarray) -> Dict:
        """Benchmark Sobel edge detection"""
        results = {}
        
        # OpenCV CPU
        if HAS_OPENCV:
            def opencv_sobel(img):
                gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
                gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
                magnitude = np.sqrt(gx**2 + gy**2)
                return np.clip(magnitude, 0, 255).astype(np.uint8)
            
            avg_time, std_time, result = self.benchmark_function(opencv_sobel, img)
            results['opencv_cpu'] = {'time': avg_time, 'std': std_time, 'result': result}
        
        # CuPy GPU
        if HAS_CUPY:
            img_gpu = cp.asarray(img, dtype=cp.float32)
            
            def cupy_sobel(img_gpu):
                sobel_x = cp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=cp.float32)
                sobel_y = cp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=cp.float32)
                
                from cupyx.scipy.ndimage import convolve
                gx = convolve(img_gpu, sobel_x)
                gy = convolve(img_gpu, sobel_y)
                magnitude = cp.sqrt(gx**2 + gy**2)
                return cp.clip(magnitude, 0, 255).astype(cp.uint8)
            
            avg_time, std_time, result = self.benchmark_function(cupy_sobel, img_gpu)
            results['cupy_gpu'] = {'time': avg_time, 'std': std_time, 'result': result}
        
        # PyTorch GPU
        if HAS_TORCH:
            img_tensor = torch.from_numpy(img).float().cuda().unsqueeze(0).unsqueeze(0)
            
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                    dtype=torch.float32, device='cuda').unsqueeze(0).unsqueeze(0)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                    dtype=torch.float32, device='cuda').unsqueeze(0).unsqueeze(0)
            
            def pytorch_sobel(img_tensor):
                padded = F.pad(img_tensor, (1, 1, 1, 1), mode='replicate')
                gx = F.conv2d(padded, sobel_x)
                gy = F.conv2d(padded, sobel_y)
                magnitude = torch.sqrt(gx**2 + gy**2)
                return torch.clamp(magnitude.squeeze(), 0, 255).byte()
            
            avg_time, std_time, result = self.benchmark_function(pytorch_sobel, img_tensor)
            results['pytorch_gpu'] = {'time': avg_time, 'std': std_time, 'result': result}
        
        # CUDA (ours)
        if HAS_CUDA_FILTERS:
            avg_time, std_time, result = self.benchmark_function(
                fastfilters.sobel, img, True
            )
            results['cuda_ours'] = {'time': avg_time, 'std': std_time, 'result': result}
        
        return results
    
    def benchmark_box_blur(self, img: np.ndarray, radius: int = 3) -> Dict:
        """Benchmark box blur"""
        results = {}
        ksize = 2 * radius + 1
        
        # OpenCV CPU
        if HAS_OPENCV:
            avg_time, std_time, result = self.benchmark_function(
                cv2.blur, img, (ksize, ksize)
            )
            results['opencv_cpu'] = {'time': avg_time, 'std': std_time, 'result': result}
        
        # CuPy GPU
        if HAS_CUPY:
            img_gpu = cp.asarray(img)
            
            def cupy_box_blur(img_gpu, ksize):
                from cupyx.scipy.ndimage import uniform_filter
                return uniform_filter(img_gpu, size=ksize).astype(cp.uint8)
            
            avg_time, std_time, result = self.benchmark_function(cupy_box_blur, img_gpu, ksize)
            results['cupy_gpu'] = {'time': avg_time, 'std': std_time, 'result': result}
        
        # CUDA (ours)
        if HAS_CUDA_FILTERS:
            avg_time, std_time, result = self.benchmark_function(
                fastfilters.box_blur, img, radius, True
            )
            results['cuda_ours'] = {'time': avg_time, 'std': std_time, 'result': result}
        
        return results
    
    def print_results(self, operation: str, results: Dict, width: int, height: int):
        """Print formatted benchmark results"""
        megapixels = (width * height) / 1e6
        
        print(f"\n{'='*80}")
        print(f"Operation: {operation}")
        print(f"Image size: {width}x{height} ({megapixels:.2f} megapixels)")
        print(f"{'='*80}")
        print(f"{'Implementation':<20} {'Time (ms)':<15} {'Throughput (MP/s)':<20} {'Speedup':<10}")
        print(f"{'-'*80}")
        
        # Get baseline (OpenCV CPU) for speedup calculation
        baseline_time = results.get('opencv_cpu', {}).get('time', None)
        
        for impl_name, data in sorted(results.items()):
            time_ms = data['time']
            throughput = megapixels / (time_ms / 1000.0)
            
            speedup_str = "-"
            if baseline_time and impl_name != 'opencv_cpu':
                speedup = baseline_time / time_ms
                speedup_str = f"{speedup:.2f}×"
            
            print(f"{impl_name:<20} {time_ms:>10.2f} ± {data['std']:>5.2f}  "
                  f"{throughput:>15.2f}  {speedup_str:>10}")
        
        print(f"{'='*80}\n")
    
    def run_full_benchmark(self):
        """Run complete benchmark suite"""
        print("="*80)
        print("CUDA Image Filters - Comprehensive Benchmark")
        print("="*80)
        
        for width, height in self.image_sizes:
            print(f"\nGenerating test image: {width}x{height}")
            img = self.generate_test_image(width, height)
            
            # Gaussian Blur
            print("\nBenchmarking Gaussian Blur...")
            gaussian_results = self.benchmark_gaussian_blur(img, sigma=2.0)
            self.print_results("Gaussian Blur (σ=2.0)", gaussian_results, width, height)
            
            # Sobel
            print("\nBenchmarking Sobel Edge Detection...")
            sobel_results = self.benchmark_sobel(img)
            self.print_results("Sobel Edge Detection", sobel_results, width, height)
            
            # Box Blur
            print("\nBenchmarking Box Blur...")
            box_results = self.benchmark_box_blur(img, radius=3)
            self.print_results("Box Blur (radius=3)", box_results, width, height)


if __name__ == "__main__":
    # Test different image sizes
    sizes = [
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
    ]
    
    bench = Benchmark(image_sizes=sizes, warmup_runs=3, benchmark_runs=10)
    bench.run_full_benchmark()
