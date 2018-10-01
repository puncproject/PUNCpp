#!/bin/bash

# USAGE:
#
#   ./install <build type>
#
# BUILD TYPES:
#
#   debug (default):
#       This compiles the fastest since it's using as little optimization
#       as possible. Suitable for everyday debugging work. It also embeds
#       instrumentation code for using the GNU Debugger (gdb).
#
#   release:
#       This runs the fastest, having maximum optimization. Suitable for
#       released versions. Should also be used when running large simulations
#       or testing performance (unless using gprof).
#
#   profile:
#       Have maximum optimization, but runs somewhat slower than release due to
#       embeded instrumentation code necessary for the GNU Profiler (gprof). 
#
#   doc:
#       Not actually a build type but generates Doxygen documentation using the
#       latest build. A link doc.html should become available.

if [ $# -eq 0 ]
then
    BUILD_TYPE="debug"
else
    # Read input into BUILD_TYPE as lowercase
    BUILD_TYPE=$(echo "$1" | sed -e 's/\(.*\)/\L\1/')

    case $BUILD_TYPE in
        debug) ;;
        release) ;;
        profile) ;;
        doc) ;;
        *)
            echo "Invalid build type: $1"
            exit 1
            ;;
    esac
fi

if [ $BUILD_TYPE = "doc" ]
then
    # Extract the most recent makefile to make documentation from
    MAKE=$(ls -t ./build/*/Makefile | grep -v latex | head -n 1)
    MAKEDIR="$(dirname "$MAKE")" 

    ln -sf build/html/index.html doc.html
    cd $MAKEDIR
    make doc
else
    echo "Compiling UFLs"

    # FIXME: Should change CMakeLists.txt to take care of this.
    cd ufl
    make -j $(nproc)
    cd ..

    echo "Building PUNC++"

    mkdir -p build/$BUILD_TYPE 
    cd build/$BUILD_TYPE
    cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE ../..
    make -j $(nproc)
    make install DESTDIR=..

    # while true
    # do
    #     read -p "Install PUNC++ to system? [y/n] " inp
    #     case $inp in
    #         [Yy]* ) sudo make install; break ;;
    #         [Nn]* ) break ;;
    #     esac
    # done

fi
