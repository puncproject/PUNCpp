cmake_minimum_required(VERSION 3.5)
set(PROJECT_NAME langmuir)
project(${PROJECT_NAME})
 
file(GLOB RUN "langmuir.cpp")

# link to punc:
set ( PROJECT_LINK_LIBS libpunc.so )
link_directories( /home/diako/Documents/cpp/punc_experimental/build )
include_directories(/home/diako/Documents/cpp/punc_experimental/punc)

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
#add_compile_options(-Wall -Wextra -Wpedantic)
CHECK_CXX_COMPILER_FLAG("-Wno-comment -O3" HAVE_NO_MULTLINE)
if (HAVE_NO_MULTLINE)
  set(CMAKE_CXX_FLAGS "-Wno-comment -O3 ${CMAKE_CXX_FLAGS}")
endif()

add_executable(${PROJECT_NAME} ${RUN})

# Target libraries
target_link_libraries(${PROJECT_NAME} ${DOLFIN_LIBRARIES}
${DOLFIN_3RD_PARTY_LIBRARIES} ${PROJECT_LINK_LIBS})
