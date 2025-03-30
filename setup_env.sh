#!/bin/bash

# Update package lists
sudo apt-get update

# Add repository for updated gcc/g++
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt-get update

# Install/Update gcc and required libraries
sudo apt-get install -y \
    gcc-11 \
    g++-11 \
    libstdc++6 \
    cmake \
    build-essential \
    libx11-dev \
    libatlas-base-dev \
    libopenblas-dev \
    python3-dev \
    python3-pip \
    libboost-all-dev \
    libgtk-3-dev

# Update alternatives to use gcc-11
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110 \
                         --slave /usr/bin/g++ g++ /usr/bin/g++-11 \
                         --slave /usr/bin/gcov gcov /usr/bin/gcov-11

# Clean and update library cache
sudo ldconfig

# Set up build environment
export CFLAGS="-I/usr/include/python3.10"
export CXXFLAGS="-I/usr/include/python3.10"

# Clean pip cache
pip cache purge

# Remove existing dlib installation
pip uninstall -y dlib

# Install dlib dependencies
pip install numpy --no-cache-dir
