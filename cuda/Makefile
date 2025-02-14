.PHONY: all clean run_cuda cmake

BUILD_DIR = build

all: cmake

cmake:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. && make

clean:
	rm -rf $(BUILD_DIR)

run_cuda: cmake
	./$(BUILD_DIR)/exec_cuda $(MAT) $(MODE)