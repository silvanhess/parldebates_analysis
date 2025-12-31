#!/bin/bash
# setup_intel_arc.sh - Setup for Intel Arc GPU

set -e

echo "=========================================="
echo "Setup for Intel Arc GPU with PyTorch"
echo "=========================================="

# Remove old environment if it exists
echo "Removing old 'thesis' environment..."
conda deactivate 2>/dev/null || true
conda env remove -n thesis -y 2>/dev/null || true

# Create fresh environment with Python 3.11
echo "Creating new environment with Python 3.11..."
conda create -n model-training python=3.11 -y

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate model-training

# Update pip
pip install --upgrade pip

# Install PyTorch with Intel XPU support
echo "Installing PyTorch with Intel XPU support..."
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install Intel Extension for PyTorch (IPEX) for Arc GPU acceleration
echo "Installing Intel Extension for PyTorch..."
python -m pip install intel-extension-for-pytorch

# Install oneCCL for distributed training (optional but recommended)
python -m pip install oneccl_bind_pt --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/

# Install core packages
echo "Installing data science packages..."
conda install pandas numpy scikit-learn -y

# Install transformers ecosystem
echo "Installing transformers..."
pip install transformers datasets tokenizers

# Install simpletransformers
echo "Installing simpletransformers..."
pip install simpletransformers

# Install wandb
echo "Installing wandb..."
pip install wandb

# Install utilities
echo "Installing utilities..."
pip install unidecode sentencepiece accelerate

# Verify installation
echo ""
echo "=========================================="
echo "Verification"
echo "=========================================="

python << 'EOF'
import sys
print(f"Python version: {sys.version}")

import torch
print(f"PyTorch version: {torch.__version__}")

# Check for Intel XPU
try:
    import intel_extension_for_pytorch as ipex
    print(f"Intel Extension for PyTorch: {ipex.__version__}")
    print(f"XPU available: {torch.xpu.is_available()}")
    if torch.xpu.is_available():
        print(f"XPU device count: {torch.xpu.device_count()}")
        print(f"XPU device name: {torch.xpu.get_device_name(0)}")
except ImportError:
    print("Intel Extension not installed (CPU mode only)")

import transformers
print(f"Transformers: {transformers.__version__}")

import pandas as pd
print(f"Pandas: {pd.__version__}")

print("\n✅ Installation complete!")
EOF

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo "Your environment is configured for Intel Arc GPU"
echo ""
echo "To activate: conda activate model-training"
