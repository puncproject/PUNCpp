#!/bin/bash
rm -rf build
mkdir -p build && cd build && cmake .. && make && ./prototype
#cd .. && python load.py
