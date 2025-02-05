cmake_minimum_required(VERSION 3.8)
project(matrix_vector_multiply CUDA CXX)

# Enable CUDA
enable_language(CUDA)
find_package(CUDA REQUIRED)

# Set compilation flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2")
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

# Set directories
include_directories(include lib)

# List source files explicitly
set(SOURCES
    src/csr.c
    src/matrix_operations.c
    src/mtx_reader.c
    src/vector_generator.c
    lib/mmio.c
)

set(CUDA_SOURCES
    src/csr_cuda.cu
    src/main.cu
)

# Create executable
add_executable(exec_cuda ${SOURCES} ${CUDA_SOURCES})

# Set CUDA properties
set_target_properties(exec_cuda PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 75)

# Link libraries
target_link_libraries(exec_cuda PRIVATE cudart m)

# Set source file properties
set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CXX)
set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)