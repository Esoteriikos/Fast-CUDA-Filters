#!/bin/bash
# Colab-specific build script
# Run this in Google Colab to build the CUDA library

set -e  # Exit on error

echo "================================"
echo "CUDA Image Filters - Colab Build"
echo "================================"

# Check CUDA
echo -e "\n📌 Checking CUDA installation..."
nvcc --version || { echo "❌ CUDA not found!"; exit 1; }
nvidia-smi || { echo "❌ No GPU found!"; exit 1; }

# Get CUDA architecture
CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
echo "✅ Detected GPU compute capability: sm_$CUDA_ARCH"

# Set paths
CUDA_HOME=/usr/local/cuda
PROJECT_ROOT=$(pwd)
SRC_DIR=$PROJECT_ROOT/src
BUILD_DIR=$PROJECT_ROOT/build
PYTHON_BINDINGS=$PROJECT_ROOT/python_bindings

echo -e "\n📦 Installing Python dependencies..."
pip install pybind11 numpy -q

echo -e "\n🔨 Creating build directory..."
mkdir -p $BUILD_DIR

echo -e "\n🔧 Compiling CUDA kernels..."

# Compile each CUDA file
for cu_file in $SRC_DIR/*.cu; do
    filename=$(basename "$cu_file")
    obj_file="$BUILD_DIR/${filename%.cu}.o"
    echo "  Compiling $filename..."
    
    nvcc --extended-lambda -c "$cu_file" -o "$obj_file" \
        -std=c++17 \
        --compiler-options '-fPIC' \
        -arch=sm_$CUDA_ARCH \
        -O3 \
        -I$CUDA_HOME/include \
        -I$SRC_DIR
done

echo -e "\n🐍 Compiling Python bindings..."

# Get Python include paths
PYTHON_INCLUDE=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")
PYBIND11_INCLUDE=$(python3 -c "import pybind11; print(pybind11.get_include())")

nvcc --extended-lambda -c $PYTHON_BINDINGS/bindings.cpp -o $BUILD_DIR/bindings.o \
    -std=c++17 \
    --compiler-options '-fPIC' \
    -I$PYBIND11_INCLUDE \
    -I$PYTHON_INCLUDE \
    -I$SRC_DIR

echo -e "\n🔗 Linking shared library..."

# Find Python extension suffix
EXT_SUFFIX=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

nvcc --shared -o $BUILD_DIR/fastfilters$EXT_SUFFIX \
    $BUILD_DIR/*.o \
    -L$CUDA_HOME/lib64 \
    -lcudart

echo -e "\n📥 Installing Python module..."

# Create __init__.py
cat > $BUILD_DIR/__init__.py << 'EOF'
"""Fast CUDA-accelerated image filters"""
from .fastfilters import *
__version__ = '0.1.0'
EOF

# Install
mkdir -p ~/.local/lib/python$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")/site-packages/fastfilters
cp $BUILD_DIR/fastfilters$EXT_SUFFIX ~/.local/lib/python$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")/site-packages/fastfilters/
cp $BUILD_DIR/__init__.py ~/.local/lib/python$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")/site-packages/fastfilters/

# Alternative: Add to Python path
export PYTHONPATH=$BUILD_DIR:$PYTHONPATH

echo -e "\n✅ Build complete!"
echo -e "\nTest the installation:"
echo "  python3 -c 'import fastfilters; print(\"✅ Import successful!\")'"

# Run quick test
echo -e "\n🧪 Running quick test..."
python3 << 'PYEOF'
import sys
sys.path.insert(0, 'build')
import fastfilters
import numpy as np

img = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
result = fastfilters.gaussian_blur(img, 2.0)
print(f"✅ Test passed! Processed {img.shape} image")
PYEOF

echo -e "\n🎉 All done! You can now use fastfilters in Python."
