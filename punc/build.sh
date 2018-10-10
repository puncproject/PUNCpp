# Builds PUNC++ and installs it locally

# Compiling UFLs
cd ufl
make -j $(nproc)
cd ..

# Building PUNC++
mkdir -p build
cd build
cmake ..
make -j $(nproc)

# Installing PUNC++ locally
mkdir -p install
make install DESTDIR=../install
