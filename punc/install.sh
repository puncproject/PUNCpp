#!/bin/bash

cd ufl
make -j $(nproc)
cd ..
# rm -rf build
# mkdir -p build 
cd build
# cmake ..
make -j $(nproc)
sudo make install
