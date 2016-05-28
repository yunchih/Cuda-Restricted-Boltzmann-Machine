# Directories
PWD       = $(shell pwd)
BUILD_DIR = $(PWD)/bin
TEST_DIR  = $(PWD)/tests
INC_DIR   = $(PWD)/include

# Flags
NVCC      = nvcc
CXXFLAGS  =
DEBUG     = -DNDEBUG #-g -DDEBUG
ARCH      = -arch=compute_30 -code=sm_30 
INC       = -I$(INC_DIR)
LIBS      = -lcublas -lcurand -lm
OPTIMIZE  = -O2
NVCCFLAGS = -ccbin="$(shell which c++)" -Xcompiler="$(CXXFLAGS)" -std=c++11 $(ARCH) $(INC) $(OPTIMIZE) $(DEBUG)

# Target
MAIN     = main.cpp
EXEC     = rbm
SOURCES  = $(wildcard *.cpp)
SOURCES += $(wildcard *.cu)
HEADERS += $(wildcard $(LIB)/*.h)
OBJECTS  = $(patsubst %.cpp,%.o,$(patsubst %.cu,%.o,$(SOURCES)))
BUILDS   = $(OBJECTS:%.o=$(BUILD_DIR)/%.o)

export # export all variables for sub-make

# Main target
all: dirs $(BUILDS)
	$(NVCC) $(LIBS) $(BUILDS) -o $(EXEC)

# To obtain object files
$(BUILD_DIR)/%.o: %.cpp $(HEADERS)
	$(NVCC) -c $(NVCCFLAGS) $< -o $@

$(BUILD_DIR)/%.o: %.cu $(HEADERS)
	$(NVCC) -c $(NVCCFLAGS) $< -o $@

.PHONY: dirs clean tests run

dirs:
	@[ -d $(BUILD_DIR) ] ||  (echo "Creating directories: $(BUILD_DIR)" && mkdir -p $(BUILD_DIR))

clean:
	rm -rf $(EXEC) $(BUILD_DIR)

tests:
	$(MAKE) -C $(TEST_DIR) bin/$(TARGET) 

run:
	# [Output directory] [Training data] [Learning rate] [Epoch number] [CD step] [Train data size] [Random sample size]
	./$(EXEC) out data/train-images-idx3-ubyte 0.02 20 10 800 10

cycle:
	$(MAKE) clean && $(MAKE) -j && $(MAKE) run
