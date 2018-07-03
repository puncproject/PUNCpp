#!/bin/bash
cd ufl
make
cd ..
rm -rf build
mkdir -p build && cd build && cmake .. && make && sudo make install
