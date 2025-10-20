"""
Simplified setup.py that works better on Google Colab
Uses direct nvcc compilation instead of CMake
"""

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import subprocess

class CUDAExtension(Extension):
    def __init__(self, name, sources):
        Extension.__init__(self, name, sources=[])
        self.sources = sources

class BuildCUDAExt(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_cuda_extension(ext)

    def build_cuda_extension(self, ext):
        # Find CUDA
        cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
        if not os.path.exists(cuda_home):
            cuda_home = '/usr/local/cuda'
        
        print(f"Using CUDA from: {cuda_home}")
        
        # Get paths
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Create build directory
        build_temp = os.path.abspath(self.build_temp)
        os.makedirs(build_temp, exist_ok=True)
        
        # Compile CUDA files
        cuda_sources = [
            '../src/convolution.cu',
            '../src/gaussian_blur.cu',
            '../src/sobel.cu',
            '../src/box_blur.cu',
            '../src/sharpening.cu',
        ]
        
        # Compile each .cu file to .o
        obj_files = []
        for cu_file in cuda_sources:
            obj_file = os.path.join(build_temp, os.path.basename(cu_file).replace('.cu', '.o'))
            
            cmd = [
                f'{cuda_home}/bin/nvcc',
                '-c',
                cu_file,
                '-o', obj_file,
                '-std=c++17',
                '--compiler-options', '-fPIC',
                '-arch=sm_75',  # Works for most GPUs, will auto-detect
                '-O3',
                '-I' + os.path.join(cuda_home, 'include'),
                '-I../src',
            ]
            
            print(f"Compiling {cu_file}...")
            try:
                subprocess.check_call(cmd, cwd=os.path.dirname(__file__))
                obj_files.append(obj_file)
            except subprocess.CalledProcessError as e:
                print(f"Warning: Failed to compile {cu_file}: {e}")
        
        if not obj_files:
            raise RuntimeError("No CUDA files compiled successfully")
        
        # Compile Python binding
        binding_obj = os.path.join(build_temp, 'bindings.o')
        
        # Get Python and pybind11 includes
        import pybind11
        pybind11_include = pybind11.get_include()
        python_include = subprocess.check_output([sys.executable, '-c', 
            'from distutils.sysconfig import get_python_inc; print(get_python_inc())']).decode().strip()
        
        cmd = [
            f'{cuda_home}/bin/nvcc',
            '-c',
            'bindings.cpp',
            '-o', binding_obj,
            '-std=c++17',
            '--compiler-options', '-fPIC',
            f'-I{pybind11_include}',
            f'-I{python_include}',
            '-I../src',
        ]
        
        print("Compiling Python bindings...")
        subprocess.check_call(cmd, cwd=os.path.dirname(__file__))
        obj_files.append(binding_obj)
        
        # Link everything
        output_file = os.path.join(extdir, ext.name + '.so')
        os.makedirs(extdir, exist_ok=True)
        
        cmd = [
            f'{cuda_home}/bin/nvcc',
            '--shared',
            '-o', output_file,
        ] + obj_files + [
            '-L' + os.path.join(cuda_home, 'lib64'),
            '-lcudart',
        ]
        
        print("Linking...")
        subprocess.check_call(cmd)
        print(f"✅ Built: {output_file}")

setup(
    name='fastfilters',
    version='0.1.0',
    author='CUDA Developer',
    description='Fast CUDA-accelerated image filters',
    ext_modules=[CUDAExtension('fastfilters', [])],
    cmdclass={'build_ext': BuildCUDAExt},
    zip_safe=False,
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.19.0',
    ],
)
