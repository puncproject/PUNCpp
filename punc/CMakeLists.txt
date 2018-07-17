cmake_minimum_required(VERSION 3.5)
project(punc)

# USE C++17

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# BUILD TYPES:

# Debug:   -Wall -O1 -g          (fastest compilation and for GNU debugger)
# Release: -Wall -O3 -NDEBUG     (fastest run-time)
# Profile: -Wall -O3 -NDEBUG -pg (for GNU profiler)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_FLAGS_DEBUG "-O1 ${CMAKE_CXX_FLAGS_DEBUG}")
set(CMAKE_CXX_FLAGS_PROFILE "-pg ${CMAKE_CXX_FLAGS_PROFILE} ${CMAKE_CXX_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS "-Wall ${CMAKE_CXX_FLAGS}")

# Source code
include_directories(ufl, include)
file(GLOB SOURCES "src/*.cpp")

# Dependencies
find_package(DOLFIN REQUIRED)
include(${DOLFIN_USE_FILE})

# Doxygen documentation
find_package(Doxygen)
if(DOXYGEN_FOUND)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile.in
        ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY
    )
    add_custom_target(doc ${DOXYGEN_EXECUTABLE}
        ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/..
        COMMENT "Generating API documentation with Doxygen" VERBATIM
    )
endif(DOXYGEN_FOUND)

# Generate the shared library from the sources
add_library(punc SHARED ${SOURCES})

# Link punc to dolfin and its 3rd party libraries
target_link_libraries(punc ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES})

# Install target
install(TARGETS punc DESTINATION /usr/lib)
