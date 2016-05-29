BIN=bin
MATRIX = $(BIN)/matrix
OUTER_PROD = $(BIN)/outer_prod
MATRIX_VEC = $(BIN)/matrix_vec
NVCCFLAGS += $(LIBS)

.PHONY: dirs
$(BIN): dirs $(MATRIX) $(OUTER_PROD) $(MATRIX_VEC)

TEST_MATRIX = matrix.cpp ../utils.cu 
$(MATRIX): $(TEST_MATRIX) dirs
	$(NVCC) $(NVCCFLAGS) $(TEST_MATRIX) -o $(MATRIX) && ./$(MATRIX)

TEST_OUTER_PROD = outer_prod.cpp ../utils.cu 
$(OUTER_PROD): $(TEST_OUTER_PROD) dirs
	$(NVCC) $(NVCCFLAGS) $(TEST_OUTER_PROD) -o $(OUTER_PROD) && ./$(OUTER_PROD)

TEST_MATRIX_VEC = matrix_vec.cpp ../utils.cu 
$(MATRIX_VEC): $(TEST_MATRIX_VEC) dirs
	$(NVCC) $(NVCCFLAGS) $(TEST_MATRIX_VEC) -o $(MATRIX_VEC) && ./$(MATRIX_VEC)

dirs:
	@[ -d $(BIN) ] ||  (echo "Creating directories" && mkdir -p $(BIN))
