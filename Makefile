CC = gcc
NVCC = nvcc
CFLAGS = -Wall -Wextra -O2 -Iinclude -Ilib
LDFLAGS = -lm

SRC_DIR = src
OBJ_DIR = obj
LIB_DIR = lib

SOURCES = $(wildcard $(SRC_DIR)/*.c) $(LIB_DIR)/mmio.c
OBJECTS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(wildcard $(SRC_DIR)/*.c)) $(OBJ_DIR)/mmio.o

EXECUTABLE = exec
OPENMP_EXECUTABLE = exec_omp
CUDA_EXECUTABLE = exec_cuda

# Controlla se nvcc (il compilatore CUDA) è installato
CUDA_AVAILABLE := $(shell command -v nvcc >/dev/null 2>&1 && echo 1 || echo 0)

# Se CUDA è disponibile, definisci la macro per la compilazione
ifeq ($(CUDA_AVAILABLE), 1)
    CFLAGS += -DCUDA_ENABLED
    CUDA_SOURCES = src/csr_cuda.cu
    CUDA_OBJECTS = $(OBJ_DIR)/csr_cuda.o
endif

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(OPENMP_EXECUTABLE): CFLAGS += -fopenmp
$(OPENMP_EXECUTABLE): LDFLAGS += -fopenmp
$(OPENMP_EXECUTABLE): $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Se CUDA è disponibile, crea anche la versione GPU
ifeq ($(CUDA_AVAILABLE),1)
$(CUDA_EXECUTABLE): $(OBJECTS) $(CUDA_OBJECTS)
	$(NVCC) -o $@ $^ -Xcompiler -fopenmp -lcudart

$(OBJ_DIR)/csr_cuda.o: $(SRC_DIR)/csr_cuda.cu | $(OBJ_DIR)
	$(NVCC) -c $< -o $@
endif

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/mmio.o: $(LIB_DIR)/mmio.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -w -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

clean:
	rm -rf $(OBJ_DIR) $(EXECUTABLE) $(OPENMP_EXECUTABLE) $(CUDA_EXECUTABLE)

run: $(EXECUTABLE)
	./$(EXECUTABLE) .matrices/$(MAT)

run_openmp: $(OPENMP_EXECUTABLE)
	./$(OPENMP_EXECUTABLE) .matrices/$(MAT)

ifdef CUDA_AVAILABLE
run_cuda: $(CUDA_EXECUTABLE)
	./$(CUDA_EXECUTABLE) .matrices/$(MAT)
endif

.PHONY: all clean run run_openmp run_cuda
