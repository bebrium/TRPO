cmake_minimum_required(VERSION 3.16)
project(TRMM_Benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenBLAS REQUIRED)
find_package(OpenMP)

add_executable(trmm_benchmark trmm_benchmark.cpp)

if(TARGET OpenBLAS::OpenBLAS)
    target_link_libraries(trmm_benchmark PRIVATE OpenBLAS::OpenBLAS)
else()
    target_include_directories(trmm_benchmark PRIVATE ${OpenBLAS_INCLUDE_DIRS})
    target_link_libraries(trmm_benchmark PRIVATE ${OpenBLAS_LIBRARIES})
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(trmm_benchmark PRIVATE OpenMP::OpenMP_CXX)
    target_compile_options(trmm_benchmark PRIVATE ${OpenMP_CXX_FLAGS})
endif()

if(MSVC)
    target_compile_options(trmm_benchmark PRIVATE /O2 /arch:AVX2)
else()
    target_compile_options(trmm_benchmark PRIVATE -O3 -march=native -flto)
endif()

message(STATUS "OpenBLAS found: ${OpenBLAS_DIR}")
message(STATUS "OpenMP enabled: ${OpenMP_CXX_FOUND}")