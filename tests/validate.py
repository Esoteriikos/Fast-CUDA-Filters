"""
Validation suite for CUDA image filters

Compares CUDA implementation outputs against OpenCV (ground truth)
- Checks numerical accuracy
- Visualizes differences
- Generates comparison plots
"""

import numpy as np
import sys
from typing import Tuple

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    print("Error: OpenCV is required for validation")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: Matplotlib not found, skipping visualizations")

try:
    import fastfilters
    HAS_CUDA_FILTERS = True
except ImportError:
    print("Error: fastfilters not found. Build Python bindings first.")
    sys.exit(1)


class Validator:
    def __init__(self, tolerance: float = 1e-2, visualize: bool = True):
        self.tolerance = tolerance
        self.visualize = visualize and HAS_MATPLOTLIB
        self.results = []
        
    def generate_test_image(self, width: int = 512, height: int = 512) -> np.ndarray:
        """Generate test image with various features"""
        img = np.zeros((height, width), dtype=np.uint8)
        
        # Gradient background
        for y in range(height):
            for x in range(width):
                img[y, x] = int((x / width) * 255)
        
        # Vertical edges
        cv2.rectangle(img, (width//4, height//4), (width//2, 3*height//4), 255, -1)
        
        # Horizontal edges
        cv2.rectangle(img, (width//2 + 20, height//3), (3*width//4, 2*height//3), 200, -1)
        
        # Circles (smooth edges)
        cv2.circle(img, (3*width//4, height//4), 40, 150, -1)
        
        # Lines (diagonal edges)
        cv2.line(img, (0, 0), (width//3, height//3), 100, 3)
        
        return img
    
    def compute_error_metrics(self, img1: np.ndarray, img2: np.ndarray) -> dict:
        """Compute various error metrics"""
        # Ensure same type
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)
        
        # Mean Absolute Error
        mae = np.mean(np.abs(img1 - img2))
        
        # Root Mean Square Error
        rmse = np.sqrt(np.mean((img1 - img2) ** 2))
        
        # Peak Signal-to-Noise Ratio
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        
        # Maximum absolute difference
        max_diff = np.max(np.abs(img1 - img2))
        
        # Percentage of pixels with difference > threshold
        diff_map = np.abs(img1 - img2)
        pct_above_threshold = np.sum(diff_map > self.tolerance) / diff_map.size * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'psnr': psnr,
            'max_diff': max_diff,
            'pct_above_tol': pct_above_threshold
        }
    
    def visualize_comparison(self, img_orig: np.ndarray, img_ref: np.ndarray, 
                            img_test: np.ndarray, title: str):
        """Create visualization comparing reference and test outputs"""
        if not self.visualize:
            return
        
        diff = np.abs(img_ref.astype(np.float32) - img_test.astype(np.float32))
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        # Original
        axes[0, 0].imshow(img_orig, cmap='gray')
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        # Reference (OpenCV)
        axes[0, 1].imshow(img_ref, cmap='gray')
        axes[0, 1].set_title('Reference (OpenCV)')
        axes[0, 1].axis('off')
        
        # Test (CUDA)
        axes[0, 2].imshow(img_test, cmap='gray')
        axes[0, 2].set_title('Test (CUDA)')
        axes[0, 2].axis('off')
        
        # Difference map
        im = axes[1, 0].imshow(diff, cmap='hot', vmin=0, vmax=10)
        axes[1, 0].set_title('Absolute Difference')
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0])
        
        # Difference histogram
        axes[1, 1].hist(diff.flatten(), bins=50, edgecolor='black')
        axes[1, 1].set_title('Difference Histogram')
        axes[1, 1].set_xlabel('Pixel Difference')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_yscale('log')
        
        # Statistics
        metrics = self.compute_error_metrics(img_ref, img_test)
        stats_text = f"MAE: {metrics['mae']:.4f}\n"
        stats_text += f"RMSE: {metrics['rmse']:.4f}\n"
        stats_text += f"PSNR: {metrics['psnr']:.2f} dB\n"
        stats_text += f"Max Diff: {metrics['max_diff']:.2f}\n"
        stats_text += f">Tol: {metrics['pct_above_tol']:.2f}%"
        
        axes[1, 2].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                       family='monospace', bbox=dict(boxstyle='round', facecolor='wheat'))
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'validation_{title.replace(" ", "_").lower()}.png', dpi=150)
        print(f"  Saved visualization: validation_{title.replace(' ', '_').lower()}.png")
        plt.close()
    
    def validate_gaussian_blur(self, img: np.ndarray, sigma: float = 2.0) -> bool:
        """Validate Gaussian blur"""
        print(f"\nValidating Gaussian Blur (sigma={sigma})...")
        
        # Reference (OpenCV)
        ksize = int(6 * sigma + 1)
        if ksize % 2 == 0:
            ksize += 1
        ref_output = cv2.GaussianBlur(img, (ksize, ksize), sigma)
        
        # Test (CUDA)
        test_output = fastfilters.gaussian_blur(img, sigma)
        
        # Compute metrics
        metrics = self.compute_error_metrics(ref_output, test_output)
        
        # Check tolerance
        passed = metrics['mae'] < self.tolerance
        
        # Print results
        print(f"  MAE: {metrics['mae']:.6f}")
        print(f"  RMSE: {metrics['rmse']:.6f}")
        print(f"  PSNR: {metrics['psnr']:.2f} dB")
        print(f"  Max Difference: {metrics['max_diff']:.2f}")
        print(f"  Pixels above tolerance: {metrics['pct_above_tol']:.2f}%")
        print(f"  Status: {'✓ PASSED' if passed else '✗ FAILED'}")
        
        self.results.append(('Gaussian Blur', passed, metrics))
        
        # Visualize
        self.visualize_comparison(img, ref_output, test_output, f"Gaussian Blur (σ={sigma})")
        
        return passed
    
    def validate_sobel(self, img: np.ndarray) -> bool:
        """Validate Sobel edge detection"""
        print(f"\nValidating Sobel Edge Detection...")
        
        # Reference (OpenCV)
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx**2 + gy**2)
        ref_output = np.clip(magnitude, 0, 255).astype(np.uint8)
        
        # Test (CUDA)
        test_output = fastfilters.sobel(img, use_accurate=True)
        
        # Compute metrics
        metrics = self.compute_error_metrics(ref_output, test_output)
        
        # More lenient for Sobel (due to different implementations)
        passed = metrics['mae'] < self.tolerance * 10
        
        print(f"  MAE: {metrics['mae']:.6f}")
        print(f"  RMSE: {metrics['rmse']:.6f}")
        print(f"  PSNR: {metrics['psnr']:.2f} dB")
        print(f"  Max Difference: {metrics['max_diff']:.2f}")
        print(f"  Pixels above tolerance: {metrics['pct_above_tol']:.2f}%")
        print(f"  Status: {'✓ PASSED' if passed else '✗ FAILED'}")
        
        self.results.append(('Sobel', passed, metrics))
        
        self.visualize_comparison(img, ref_output, test_output, "Sobel Edge Detection")
        
        return passed
    
    def validate_box_blur(self, img: np.ndarray, radius: int = 3) -> bool:
        """Validate box blur"""
        print(f"\nValidating Box Blur (radius={radius})...")
        
        ksize = 2 * radius + 1
        
        # Reference (OpenCV)
        ref_output = cv2.blur(img, (ksize, ksize))
        
        # Test (CUDA)
        test_output = fastfilters.box_blur(img, radius, use_separable=True)
        
        # Compute metrics
        metrics = self.compute_error_metrics(ref_output, test_output)
        
        passed = metrics['mae'] < self.tolerance
        
        print(f"  MAE: {metrics['mae']:.6f}")
        print(f"  RMSE: {metrics['rmse']:.6f}")
        print(f"  PSNR: {metrics['psnr']:.2f} dB")
        print(f"  Max Difference: {metrics['max_diff']:.2f}")
        print(f"  Pixels above tolerance: {metrics['pct_above_tol']:.2f}%")
        print(f"  Status: {'✓ PASSED' if passed else '✗ FAILED'}")
        
        self.results.append(('Box Blur', passed, metrics))
        
        self.visualize_comparison(img, ref_output, test_output, f"Box Blur (radius={radius})")
        
        return passed
    
    def validate_custom_kernel(self, img: np.ndarray) -> bool:
        """Validate custom convolution"""
        print(f"\nValidating Custom Convolution...")
        
        # Edge detection kernel
        kernel = np.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]], dtype=np.float32)
        
        # Reference (OpenCV)
        ref_output = cv2.filter2D(img, -1, kernel)
        
        # Test (CUDA)
        test_output = fastfilters.convolve2d(img, kernel)
        
        # Compute metrics
        metrics = self.compute_error_metrics(ref_output, test_output)
        
        passed = metrics['mae'] < self.tolerance * 5  # More lenient
        
        print(f"  MAE: {metrics['mae']:.6f}")
        print(f"  RMSE: {metrics['rmse']:.6f}")
        print(f"  PSNR: {metrics['psnr']:.2f} dB")
        print(f"  Max Difference: {metrics['max_diff']:.2f}")
        print(f"  Pixels above tolerance: {metrics['pct_above_tol']:.2f}%")
        print(f"  Status: {'✓ PASSED' if passed else '✗ FAILED'}")
        
        self.results.append(('Custom Convolution', passed, metrics))
        
        self.visualize_comparison(img, ref_output, test_output, "Custom Convolution")
        
        return passed
    
    def run_all_validations(self):
        """Run all validation tests"""
        print("="*80)
        print("CUDA Image Filters - Validation Suite")
        print("="*80)
        
        # Generate test image
        print("\nGenerating test image...")
        img = self.generate_test_image(512, 512)
        
        # Run validations
        all_passed = True
        all_passed &= self.validate_gaussian_blur(img, sigma=1.5)
        all_passed &= self.validate_gaussian_blur(img, sigma=3.0)
        all_passed &= self.validate_sobel(img)
        all_passed &= self.validate_box_blur(img, radius=3)
        all_passed &= self.validate_box_blur(img, radius=5)
        all_passed &= self.validate_custom_kernel(img)
        
        # Summary
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        
        for test_name, passed, metrics in self.results:
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"{test_name:<30} {status:>15}  (MAE: {metrics['mae']:.6f})")
        
        print("="*80)
        
        if all_passed:
            print("\n✓ All tests PASSED!")
            return 0
        else:
            print("\n✗ Some tests FAILED!")
            return 1


if __name__ == "__main__":
    validator = Validator(tolerance=1.0, visualize=True)
    exit_code = validator.run_all_validations()
    sys.exit(exit_code)
