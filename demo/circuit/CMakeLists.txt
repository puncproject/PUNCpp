cmake_minimum_required(VERSION 3.5)
set(PROJECT_NAME circuit)
project(${PROJECT_NAME})

# Default to Release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)

add_compile_options(-Wall)

file(GLOB RUN "circuit.cpp")

list(APPEND CMAKE_FIND_ROOT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../punc/install")
list(APPEND CMAKE_FIND_ROOT_PATH "~/.conda/envs/")

find_library(PUNC punc)
find_path(PUNC_INCLUDE_DIR punc)
include_directories(${PUNC_INCLUDE_DIR})
message(STATUS "PUNC dir: ${PUNC}")
message(STATUS "PUNC include dir: ${PUNC_INCLUDE_DIR}")

find_package(DOLFIN REQUIRED)
include(${DOLFIN_USE_FILE})

find_package(Boost COMPONENTS program_options timer chrono REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
message(STATUS "Boost libraries: ${Boost_LIBRARIES}")
message(STATUS "Boost include dir: ${Boost_INCLUDE_DIRS}")

add_executable(${PROJECT_NAME} ${RUN})
target_link_libraries(${PROJECT_NAME} LINK_PUBLIC ${PUNC} ${DOLFIN_LIBRARIES}
    ${DOLFIN_3RD_PARTY_LIBRARIES} ${Boost_LIBRARIES})
