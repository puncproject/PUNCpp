#!/bin/bash
rm -rf build
mkdir -p build && cd build && cmake .. && make && mpirun -np 4 ./par
#cd .. && python plot.py

