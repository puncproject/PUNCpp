#!/bin/bash
rm -rf build
mkdir -p build && cd build && cmake .. && make && ./basis
#cd .. && python load.py
