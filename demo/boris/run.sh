#!/bin/bash
rm -rf build
mkdir -p build && cd build && cmake .. && make && ./boris
cd .. && python particle.py

