#!/bin/bash
rm -rf build
mkdir -p build && cd build && cmake .. && make && ./convergence
cd .. && python plot.py

