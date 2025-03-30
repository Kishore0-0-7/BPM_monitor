#!/bin/bash

# Run libstdc++ fix first
./fix_libstdcpp.sh

# Remove any existing dlib installation
pip uninstall -y dlib face-recognition-models

# Install build dependencies first
sudo apt-get update && sudo apt-get install -y \
    cmake \
    build-essential \
    libx11-dev \
    libatlas-base-dev

# Set compiler flags
export CFLAGS="-O3"
export CXXFLAGS="-O3"

# Install dlib using requirements file
pip install -r dlib-requirements.txt

echo "dlib installation completed"
