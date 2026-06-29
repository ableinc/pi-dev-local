#!/bin/bash
# Get program arguments
LLAMA_CPP_PATH="$1"
if [ -z "$LLAMA_CPP_PATH" ]; then
  echo "You must provide the llama.cpp source path"
  exit 1
fi
if [ ! -d "$LLAMA_CPP_PATH" ]; then
  echo "$LLAMA_CPP_PATH is not a directory or does not exist"
  exit 1
fi
cd $LLAMA_CPP_PATH
echo "Removing old llama.cpp build path: $LLAMA_CPP_PATH/build (CTRL+C to stop)..."
sleep 3
# Clear any old configurations to prevent cache conflicts
if [ -d "build" ]; then
  rm -rf build
fi
# Create new directory
mkdir build
# Configure the optimized build
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DGGML_BUILD_REALS_NCCL=ON \
  -DCMAKE_C_FLAGS="-march=native" \
  -DCMAKE_CXX_FLAGS="-march=native"
# Compile (AMD Ryzen 9 7950X3D 16 Core/ 32 Thread) using 16 cores (half of the processing cores)
# Run nproc or lscpu (detailed) to get your cpu count
cmake --build build --config Release -j 16
echo "build-llamacpp-native.sh script is complete"
echo "Note: To install the binaries to your system-wide directory run - cmake --install $LLAMA_CPP_PATH/build"
