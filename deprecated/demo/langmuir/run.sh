#!/bin/bash
rm -rf build
mkdir -p build && cd build && cmake .. && make && ./langmuir 
cd .. && python plot.py
