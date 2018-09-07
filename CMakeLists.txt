cmake_minimum_required(VERSION 3.0)

FILE(GLOB SRC_FILES *.cpp hw/*.cpp)

project(Hetero-KMeans)
add_executable(kmeans.elf ${SRC_FILES})
include_directories("include")

install(TARGETS kmeans.elf RUNTIME DESTINATION bin)

