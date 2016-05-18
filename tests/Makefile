BIN=bin
MATRIX = $(BIN)/matrix
OUTER_PROD = $(BIN)/outer_prod

.PHONY: dirs
$(BIN): dirs $(MATRIX) $(OUTER_PROD)

TEST_MATRIX = matrix.cpp ../utils.cu 
$(MATRIX): $(TEST_MATRIX)
	$(NVCC) $(NVCCFLAGS) $(TEST_MATRIX) -o $(MATRIX) && ./$(MATRIX)

TEST_OUTER_PROD = outer_prod.cpp ../utils.cu 
$(OUTER_PROD): $(TEST_OUTER_PROD)
	$(NVCC) $(NVCCFLAGS) $(TEST_OUTER_PROD) -o $(OUTER_PROD) && ./$(OUTER_PROD)

dirs:
	@[ -d $(BIN) ] ||  (echo "Creating directories" && mkdir -p $(BIN))
