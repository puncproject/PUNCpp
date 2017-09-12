#!/bin/bash

rm -rf build
mkdir -p build && cd build && cmake .. && make && ./demo && cd .. && python plot.py
