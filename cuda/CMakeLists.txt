cmake_minimum_required(VERSION 3.8)

# Versioni consigliate per CUDA

# Versione CUDA	    Versione minima CMake
# CUDA 9	            3.8
# CUDA 10	            3.10
# CUDA 11	            3.17
# CUDA 12	            3.20+

# Se non sai che versione di Cmake hai:
# cmake --version

project(parallel_matrix_vector CUDA CXX)

# Abilita il supporto per CUDA in CMake
enable_language(CUDA)

# Cerca e trova il toolkit di CUDA installato sul sistema
find_package(CUDA REQUIRED)

# Imposta lo standard di C e CUDA
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Imposta i flag di compilazione per C e CUDA
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# La directory per gli header
include_directories(lib)

# Definizione dei file sorgente in C
set(SOURCES
    src/csr.c
    src/ellpack.c
    src/hll.c
    src/mtx_reader.c
    src/vector_generator.c
    lib/mmio.c
)

# Definizione dei file CUDA
set(CUDA_SOURCES
    src/res_utils.cu
    src/csr_cuda.cu
    src/main.cu
)

# Creazione dell'eseguibile
add_executable(exec_cuda ${SOURCES} ${CUDA_SOURCES})

# Impostazioni specifiche per CUDA
set_target_properties(exec_cuda PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON   # Permette la compilazione separata dei file CUDA
    CUDA_ARCHITECTURES 75       # Target architettura CUDA
    # Il numero 75 si riferisce all'architettura compute capability della GPU.
)

# Link alle librerie necessarie
target_link_libraries(exec_cuda PRIVATE cudart m)

# Specifica il linguaggio per i file sorgente
set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CXX)
set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)