#!/bin/bash

# make meshes
make

rm -rf build
mkdir -p build && cd build && cmake .. && make
cd ..
cp build/injection . 
./injection
python particleNumber.py && python hist.py
