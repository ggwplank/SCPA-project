CC = gcc
CFLAGS = -Wall -Wextra -O3 -Ilib
LDFLAGS = -lm

SRC_DIR = src
OBJ_DIR = obj
LIB_DIR = lib

SOURCES = $(wildcard $(SRC_DIR)/*.c) $(LIB_DIR)/mmio.c
OBJECTS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(wildcard $(SRC_DIR)/*.c)) $(OBJ_DIR)/mmio.o

EXECUTABLE = exec
OPENMP_EXECUTABLE = exec_omp

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(OPENMP_EXECUTABLE): CFLAGS += -fopenmp
$(OPENMP_EXECUTABLE): LDFLAGS += -fopenmp
$(OPENMP_EXECUTABLE): $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/mmio.o: $(LIB_DIR)/mmio.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -w -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

clean:
	rm -rf $(OBJ_DIR) $(EXECUTABLE) $(OPENMP_EXECUTABLE)

run: $(EXECUTABLE)
	./$(EXECUTABLE) ../.matrices/$(MAT) $(MODE) 

run_openmp: $(OPENMP_EXECUTABLE)
	./$(OPENMP_EXECUTABLE) ../.matrices/$(MAT) $(MODE) $(THREADS)

.PHONY: all clean