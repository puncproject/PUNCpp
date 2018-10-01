#!/bin/bash
rm -rf build
mkdir -p build && cd build && cmake .. && make && ./varobj
#cd .. && python load.py
