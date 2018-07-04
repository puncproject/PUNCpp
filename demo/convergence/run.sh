#!/bin/bash
rm -rf build
mkdir -p build && cd build && cmake .. && make
cd ..
cp build/convergence .
mpirun -np 4 ./convergence
python plot.py

