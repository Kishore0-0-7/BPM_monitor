#!/bin/bash

# Update and add required repository
sudo apt-get update
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt-get update

# Install latest GCC and G++
sudo apt-get install -y gcc-12 g++-12

# Install the required libstdc++
sudo apt-get install -y libstdc++6

# Create symbolic links if needed
sudo ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /usr/lib/libstdc++.so.6

# Update library cache
sudo ldconfig

echo "libstdc++ update completed"
