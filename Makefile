CC = gcc
CFLAGS = -Wall -Wextra -O2 -Iinclude -Ilib
LDFLAGS = -lm

SRC_DIR = src
OBJ_DIR = obj
LIB_DIR = lib

# Prende tutti i file .c da src/ e aggiunge manualmente mmio.c da lib/
SOURCES = $(wildcard $(SRC_DIR)/*.c) $(LIB_DIR)/mmio.c
OBJECTS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(wildcard $(SRC_DIR)/*.c)) $(OBJ_DIR)/mmio.o

EXECUTABLE = exec

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/mmio.o: $(LIB_DIR)/mmio.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

clean:
	rm -rf $(OBJ_DIR) $(EXECUTABLE)

run: $(EXECUTABLE)
	./$(EXECUTABLE) $(MAT)

.PHONY: all clean