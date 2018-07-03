#!/bin/bash
rm -rf build
mkdir -p build && cd build && cmake .. && make && mpirun -np 4 ./convergence
cd .. && python plot.py

