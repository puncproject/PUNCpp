mkdir -p build
cd build
# cmake -DBOOST_ROOT=~/.conda/envs/fenics ..
cmake ..
make
cd ..
ln -sf build/interaction
