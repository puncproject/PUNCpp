#!/bin/bash
cd ufl
bash compileUFL.sh
cd ..
rm -rf build
mkdir -p build && cd build && cmake .. && make && sudo make install
