# Builds interaction

mkdir -p build
cd build
cmake ..
make
cd ..
ln -sf build/interaction
