from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import subprocess

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build this extension")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            '-DBUILD_TESTS=OFF',
        ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        cmake_args += [f'-DCMAKE_BUILD_TYPE={cfg}']
        
        # Parallel build
        build_args += ['--', '-j4']

        env = os.environ.copy()
        
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        
        # CMake configure
        subprocess.check_call(
            ['cmake', ext.sourcedir] + cmake_args, 
            cwd=self.build_temp, 
            env=env
        )
        
        # Build
        subprocess.check_call(
            ['cmake', '--build', '.'] + build_args, 
            cwd=self.build_temp
        )

setup(
    name='fastfilters',
    version='0.1.0',
    author='CUDA Developer',
    author_email='',
    description='Fast CUDA-accelerated image filters',
    long_description='',
    ext_modules=[CMakeExtension('fastfilters')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.19.0',
    ],
    extras_require={
        'dev': [
            'opencv-python',
            'cupy-cuda12x',  # Adjust for your CUDA version
            'torch',
            'matplotlib',
        ],
    },
)
