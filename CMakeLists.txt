cmake_minimum_required(VERSION 3.5)

set(PROJECT_NAME demo)
project(${PROJECT_NAME})

#file(GLOB SOURCES "punc/*.cpp")
#file(GLOB HEADERS "punc/*.h")
#include_directories(punc)
#file(GLOB helper "punc/circuit.xml")

include_directories("${CMAKE_SOURCE_DIR}/punc")

file(GLOB RUN "demo_poisson.cpp")
#file(GLOB RUN "demo_object.cpp")

set ( SOURCES
 create.cpp
 punc/object.cpp
 punc/poisson.cpp
 punc/capacitance.cpp
)

set ( HEADERS
 create.h
 punc/Potential.h
 punc/EField.h
 punc/Flux.h
 punc/Error.h
 punc/Errorv2.h
 punc/object.h
 punc/poisson.h
 punc/capacitance.h
)

# Set CMake behavior
cmake_policy(SET CMP0004 NEW)

# Get DOLFIN configuration data (DOLFINConfig.cmake must be in
# DOLFIN_CMAKE_CONFIG_PATH)
find_package(DOLFIN REQUIRED)

include(${DOLFIN_USE_FILE})

# Default build type (can be overridden by user)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
      "Choose the type of build, options are: Debug MinSizeRel Release RelWithDebInfo." FORCE)
endif()

# Do not throw error for 'multi-line comments' (these are typical in
# rst which includes LaTeX)
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-Wno-comment" HAVE_NO_MULTLINE)
if (HAVE_NO_MULTLINE)
  set(CMAKE_CXX_FLAGS "-Wno-comment ${CMAKE_CXX_FLAGS}")
endif()


add_executable(${PROJECT_NAME} ${RUN} ${SOURCES} ${HEADERS})

# Target libraries
target_link_libraries(${PROJECT_NAME} dolfin)
