# Directories & files
PWD       := $(shell pwd)
NOW       := $(shell date +%Y-%m-%d_%H:%M:%S)
BUILD_DIR := $(PWD)/bin
TEST_DIR  := $(PWD)/tests
INC_DIR   := $(PWD)/include
OUT_DIR   := $(PWD)/$(NOW)_out
TILER     := montage
COMBINED  := $(NOW)_combined.pgm

# Flags
NVCC      = nvcc
CXXFLAGS  = -fopenmp
DEBUG     = -DNDEBUG # -g -DDEBUG
ARCH      = -arch=compute_30 -code=sm_30 
INC       = -I$(INC_DIR)
LIBS      = -lcublas -lcurand -lm -lgomp -lpthread
OPTIMIZE  = -O2
NVCCFLAGS = -ccbin="$(shell which c++)" -Xcompiler="$(CXXFLAGS)" -std=c++11 $(ARCH) $(INC) $(OPTIMIZE) $(DEBUG)
NVCCLINK  = $(LIBS) -Xcompiler="$(CXXFLAGS)"

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
	$(NVCC) -o $(EXEC) $(NVCCLINK) $(BUILDS)

# To obtain object files
$(BUILD_DIR)/%.o: %.cpp $(HEADERS)
	$(NVCC) -c $(NVCCFLAGS) $< -o $@

$(BUILD_DIR)/%.o: %.cu $(HEADERS)
	$(NVCC) -c $(NVCCFLAGS) $< -o $@

.PHONY: dirs clean tests run runall

dirs:
	@[ -d $(BUILD_DIR) ] ||  (echo "Creating directories: $(BUILD_DIR)" && mkdir -p $(BUILD_DIR))

clean:
	rm -rf $(EXEC) $(BUILD_DIR)

tests:
	$(MAKE) -C $(TEST_DIR) bin/$(TARGET) 

run:
	# [Output directory] [Train filename] [Test filename] [Learning rate] [Epoch number] [CD step] [Train data size] [Test data size] [Random sample size]
	./$(EXEC) $(OUT_DIR) data/train-images-idx3-ubyte data/t10k-images-idx3-ubyte 0.02 40 1 1000 200 10

time:
	time $(MAKE) run

runall: run
	@which $(TILER) 2>&1 > /dev/null \
		&& $(TILER) -resize 300% $(OUT_DIR)/*  -pointsize 14 -set label "%f" -geometry '+5+5>' $(COMBINED) \
		&& rm -rf $(OUT_DIR) \
		|| echo "Please ensure ImageMagick is installed and the arguments passwd to \"$(TILER)\" are correct";

cycle:
	$(MAKE) clean && $(MAKE) -j && $(MAKE) run
