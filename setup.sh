#!/bin/bash
# ===============================================================
# Setup - High Performance MoE Environment
# ===============================================================
# This script sets up a modern MoE development environment:
# - Verifies GPU & CUDA
# - Installs system dependencies
# - Upgrades CMake if needed
# - Sets up Python venv
# - Installs CUDA-optimized PyTorch
# - Compiles Microsoft Tutel from source
# - Verifies hardware and environment
# ===============================================================

set -e  # Exit immediately on any error

echo "🚀 Initializing Quantitative Environment Setup..."

# ===============================================================
# 1️⃣ Hardware Verification
# Check for CUDA-capable GPU via nvidia-smi
# ===============================================================
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ ERROR: nvidia-smi not found. CUDA GPU is required for Tutel MoE."
    exit 1
fi
echo "✅ CUDA-capable GPU detected."

# ===============================================================
# 2️⃣ System Dependencies & Toolchain
# Install build-essential, OpenMP, wget, Python dev tools
# ===============================================================
echo "📦 Installing base system dependencies..."
sudo apt-get update
sudo apt-get install -y build-essential libomp-dev wget gpg python3-venv python3-pip python3-dev

# ===============================================================
# 3️⃣ Modernize CMake
# Tutel requires >=3.18; Ubuntu default may be older
# ===============================================================
CURRENT_CMAKE_VER=$(cmake --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "0.0")
if $(awk 'BEGIN {exit !('"$CURRENT_CMAKE_VER"' < 3.18)}'); then
    echo "📦 Upgrading CMake to >=3.18 from Kitware..."
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null \
        | gpg --dearmor - \
        | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
    echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main' \
        | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null
    sudo apt-get update
    sudo apt-get install -y cmake
else
    echo "✅ CMake version $CURRENT_CMAKE_VER is sufficient."
fi

# Install NCCL for multi-GPU communications
sudo apt-get install -y libnccl-dev

# ===============================================================
# 4️⃣ Python Virtual Environment Setup
# ===============================================================
echo "🐍 Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

echo "⬆️ Upgrading pip, setuptools, wheel, and ninja..."
pip install --upgrade pip setuptools wheel ninja

# ===============================================================
# 5️⃣ Install CUDA-Optimized PyTorch
# ===============================================================
echo "🔥 Installing PyTorch with CUDA 12.1..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ===============================================================
# 6️⃣ Build and Install Microsoft Tutel (MoE Gateway)
# ===============================================================
echo "🧠 Compiling Microsoft Tutel from source (custom CUDA kernels)..."
pip install -v -U --no-build-isolation git+https://github.com/microsoft/tutel@main

# ===============================================================
# 7️⃣ Install Additional Python Dependencies
# ===============================================================
echo "📦 Installing other required Python modules..."
pip install -r requirements.txt

# ===============================================================
# 8️⃣ Verification
# Run Tutel Hello World to verify GPU/CUDA compilation
# ===============================================================
echo "🧪 Running Tutel Hardware Sympathy Check..."
if python3 -m tutel.examples.helloworld --batch_size=16; then
    echo "✅ MoE Environment is LIVE."
    echo "👉 Next step: source venv/bin/activate to start working."
else
    echo "❌ ERROR: Tutel validation failed. Check CUDA compilation logs above."
    exit 1
fi