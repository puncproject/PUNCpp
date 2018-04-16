cmake_minimum_required(VERSION 3.5)

set(PROJECT_NAME laframboise)
project(${PROJECT_NAME})

include_directories("${CMAKE_SOURCE_DIR} ../../punc")

file(GLOB RUN "laframboise.cpp")

set ( SOURCES
  ../../punc/object.cpp
  ../../punc/poisson.cpp
  ../../punc/injector.cpp
  ../../punc/population.cpp
  ../../punc/diagnostics.cpp
  ../../punc/distributor.cpp
  ../../punc/pusher.cpp
)

set ( HEADERS
  ../../punc/Potential1D.h
  ../../punc/Potential2D.h
  ../../punc/Potential3D.h    
  ../../punc/EField1D.h
  ../../punc/EField2D.h
  ../../punc/EField3D.h    
  ../../punc/Flux.h
  ../../punc/Energy.h
  ../../punc/object.h
  ../../punc/poisson.h
  ../../punc/injector.h
  ../../punc/population.h
  ../../punc/diagnostics.h
  ../../punc/distributor.h
  ../../punc/pusher.h
)

# Set CMake behavior
cmake_policy(SET CMP0004 NEW)

# Get DOLFIN configuration data (DOLFINConfig.cmake must be in
# DOLFIN_CMAKE_CONFIG_PATH)
find_package(DOLFIN REQUIRED)

include(${DOLFIN_USE_FILE})

# Find mshr
#find_library(MSHR_LIBRARY mshr)

# Default build type (can be overridden by user)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
      "Choose the type of build, options are: Debug MinSizeRel Release RelWithDebInfo." FORCE)
endif()

# Do not throw error for 'multi-line comments' (these are typical in
# rst which includes LaTeX)
include(CheckCXXCompilerFlag)
#add_compile_options(-Wall -Wextra -Wpedantic)
CHECK_CXX_COMPILER_FLAG("-Wno-comment -O3" HAVE_NO_MULTLINE)
if (HAVE_NO_MULTLINE)
  set(CMAKE_CXX_FLAGS "-Wno-comment -O3 ${CMAKE_CXX_FLAGS}")
endif()


add_executable(${PROJECT_NAME} ${RUN} ${SOURCES} ${HEADERS})

# Target libraries
target_link_libraries(${PROJECT_NAME} ${DOLFIN_LIBRARIES}
${DOLFIN_3RD_PARTY_LIBRARIES})
