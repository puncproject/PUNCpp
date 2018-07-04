#!/bin/bash
rm -rf build
mkdir -p build && cd build && cmake .. && make 
cd ..
cp build/boris .
./boris
python particle.py

