cmake_minimum_required(VERSION 3.5)

project(punc)
set(CMAKE_BUILD_TYPE Release)

# Use C++11
set(CMAKE_CXX_STANDARD 11)

# Require C++11
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(ufl,include)

file(GLOB SOURCES "src/*.cpp")

# dolfin
find_package(DOLFIN REQUIRED)
include(${DOLFIN_USE_FILE})

# Generate the shared library from the sources
add_library(punc SHARED ${SOURCES})

# Link punc to dolfin and its 3rd party libraries
target_link_libraries(punc ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES})

install(TARGETS punc DESTINATION /usr/lib)
