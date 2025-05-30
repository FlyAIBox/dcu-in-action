#!/bin/bash

# DCU-in-Action Quick Start Script
# This script sets up the development environment for DCU projects

set -e

echo "=========================================="
echo "DCU-in-Action Quick Start Setup"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_warning "This script is running as root. Consider running as a regular user."
fi

# Check system requirements
print_status "Checking system requirements..."

# Check Python version
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    print_status "Python version: $PYTHON_VERSION"
    
    # Check if Python version is >= 3.8
    if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)'; then
        print_status "Python version is compatible (>= 3.8)"
    else
        print_error "Python version must be >= 3.8"
        exit 1
    fi
else
    print_error "Python3 is not installed"
    exit 1
fi

# Check pip
if command -v pip3 &> /dev/null; then
    print_status "pip3 is available"
else
    print_error "pip3 is not installed"
    exit 1
fi

# Check if we're in the correct directory
if [[ ! -f "requirements.txt" ]]; then
    print_error "requirements.txt not found. Please run this script from the project root directory."
    exit 1
fi

# Install Python dependencies
print_status "Installing Python dependencies..."
pip3 install -r requirements.txt

# Check ROCm installation (optional)
print_status "Checking ROCm installation..."
if command -v rocm-smi &> /dev/null; then
    print_status "ROCm SMI is available"
    rocm-smi --version | head -1
else
    print_warning "ROCm SMI not found. DCU monitoring features will be limited."
fi

# Check PyTorch ROCm support
print_status "Checking PyTorch ROCm support..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'Device {i}: {torch.cuda.get_device_name(i)}')
"

# Test DCU manager
print_status "Testing DCU manager..."
python3 -c "
from common.dcu import DCUManager
dcu = DCUManager()
print(f'DCU available: {dcu.is_available()}')
print(f'Device count: {dcu.get_device_count()}')
"

# Create necessary directories
print_status "Creating project directories..."
mkdir -p logs
mkdir -p outputs
mkdir -p checkpoints
mkdir -p data

# Set permissions
chmod +x scripts/setup/*.sh
chmod +x scripts/deployment/*.sh 2>/dev/null || true
chmod +x scripts/monitoring/*.sh 2>/dev/null || true

# Success message
print_status "Quick start setup completed successfully!"
echo ""
echo "=========================================="
echo "Next steps:"
echo "1. Check examples/ directory for tutorials"
echo "2. Run 'make test' to verify installation"
echo "3. See README.md for detailed documentation"
echo "==========================================" 