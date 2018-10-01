#!/bin/bash
rm -rf build
mkdir -p build && cd build && cmake .. && make 
cd ..
cp build/langmuir . 
./langmuir
python plot.py
