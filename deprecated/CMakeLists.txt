cmake_minimum_required(VERSION 3.5)

project(punc)
set(CMAKE_BUILD_TYPE Release)
 
include_directories(punc)
 
file(GLOB SOURCES "punc/*.cpp")
 
# dolfin
find_package(DOLFIN REQUIRED)
include(${DOLFIN_USE_FILE})

#Generate the shared library from the sources
add_library(punc SHARED ${SOURCES})
 
install(TARGETS punc DESTINATION /usr/lib)
