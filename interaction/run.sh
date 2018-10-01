#!/bin/bash
# rm -rf build
mkdir -p build && cd build && cmake .. && make 
cd ..
ln -sf build/interaction .
./interaction "$@"
