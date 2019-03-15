# Very crude integration test. Runs a simulation that is known to have a small
# relative error and checks that this is indeed true. Since PUNC++
# unfortunately do not have an option to use the same seed each time, it is
# theoretically possible that this test fails when nothing is wrong. If it
# fails, try again. If it fails again, something is wrong.
#
# For Anaconda users: the makefile assumes the dolfin-convert and gmsh commands
# to both work from the current environment. If that's not the case, bypass
# this makefile by generating the .xml file yourself.

# Make meshes
make

# Build PUNC
cd ../punc
./build.sh

# Build Interaction
cd ../interaction
./build.sh

# Run simulation
cd ../test
rm -f *.dat
ln -sf ../interaction/build/interaction
./interaction test.ini

# Test simulation results 
./test.py
