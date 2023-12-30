
cmake_minimum_required(VERSION 3.1)
project(
  testtryfinally
  VERSION 1.0
  LANGUAGES CXX)

set(Tensorflow_IncludeDir "")

add_executable(testtryfinally "main.cpp")

include_directories("libs")
file(GLOB source_libs "libs/*.h" "libs/*.cpp")
file(GLOB source_libs_cloud "libs/cloud/*.h" "libs/cloud/*.cpp")
file(GLOB source_libs_tensorflow "libs/tensorflow/*.h" "libs/tensorflow/*.cpp")
add_library(sharedlibs SHARED ${source_libs} ${source_libs_cloud} ${source_libs_tensorflow})

target_link_libraries(testtryfinally PRIVATE sharedlibs tensorflow)

file(GLOB HEADER_FILES_libs "libs/*.h")
file(GLOB HEADER_FILES_libs_cloud "libs/cloud/*.h")

file(COPY ${HEADER_FILES_libs} DESTINATION ${CMAKE_BINARY_DIR}/libs)
file(COPY ${HEADER_FILES_libs_cloud} DESTINATION ${CMAKE_BINARY_DIR}/libs/cloud)

message("CMAKE_SOURCE_DIR" ${CMAKE_SOURCE_DIR})
message("CMAKE_BINARY_DIR" ${CMAKE_BINARY_DIR})
