cmake_minimum_required(VERSION 3.5)
project(punc)

# Default to Release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)

# Compile options
add_compile_options(-Wall)
#add_compile_options(-Wextra)

# Source code
include_directories(ufl, include)
file(GLOB SOURCE_FILES "src/*.cpp")
file(GLOB HEADER_FILES_MAIN "include/*.h")
file(GLOB HEADER_FILES_SECONDARY "include/punc/*.h")

# Find Dolfin
find_package(DOLFIN REQUIRED)
include(${DOLFIN_USE_FILE})

# Find Doxygen
find_package(Doxygen)
if(DOXYGEN_FOUND)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile.in
        ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY
    )
    add_custom_target(doc ${DOXYGEN_EXECUTABLE}
        ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/..
        COMMENT "Generating documentation using Doxygen" VERBATIM
    )
endif(DOXYGEN_FOUND)

# Add target library (punc.so), include headers and link with libraries
add_library(punc SHARED ${SOURCE_FILES})
target_link_libraries(punc ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES})
set_target_properties(PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

# Install target
install(TARGETS punc
        PUBLIC_HEADER DESTINATION include/punc   # .h files
        LIBRARY DESTINATION lib                  # .so files
        ARCHIVE DESTINATION lib                  # .a files
        RUNTIME DESTINATION bin)                 # executables

# For some reason the above doesn't install header files. Do it manually.
install(FILES ${HEADER_FILES_MAIN}
        DESTINATION include)
install(FILES ${HEADER_FILES_SECONDARY}
        DESTINATION include/punc)
