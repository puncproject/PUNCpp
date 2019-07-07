#!/bin/bash

# Make meshes
make 

# Run capacitance test
rm -rf build
mkdir -p build && cd build && cmake .. && make
cd ..
cp build/capacitance .
./capacitance

