# Makefile for Hybrid Parallel BFS Project
#
# Requirements:
#   - MPI compiler wrapper (mpicxx / mpic++)
#   - OpenMP support (-fopenmp / /openmp)
#
# Targets:
#   make all          — build bfs.exe and graph_gen.exe
#   make bfs          — build BFS executable only
#   make graph_gen    — build graph generator only
#   make clean        — remove binaries and object files
#   make graphs       — generate test graphs (20K–160K vertices)

CXX      ?= mpicxx
CXXFLAGS ?= -O3 -std=c++17 -fopenmp -Wall -Wextra
LDFLAGS  ?= -fopenmp

SRC_DIR  = src
BIN_DIR  = bin
OBJ_DIR  = build

# Source files
GRAPH_GEN_SRC = $(SRC_DIR)/graph_generator.cpp $(SRC_DIR)/graph_io.cpp
BFS_SRC       = $(SRC_DIR)/main.cpp $(SRC_DIR)/bfs_mpi.cpp \
                $(SRC_DIR)/bfs_hybrid.cpp $(SRC_DIR)/graph_io.cpp

# Object files
GRAPH_GEN_OBJ = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(GRAPH_GEN_SRC))
BFS_OBJ       = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(BFS_SRC))

# Targets
.PHONY: all bfs graph_gen clean graphs dirs

all: dirs $(BIN_DIR)/bfs.exe $(BIN_DIR)/graph_gen.exe

dirs:
	@mkdir -p $(BIN_DIR) $(OBJ_DIR)

bfs: dirs $(BIN_DIR)/bfs.exe

graph_gen: dirs $(BIN_DIR)/graph_gen.exe

# Link BFS executable
$(BIN_DIR)/bfs.exe: $(BFS_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Link graph generator executable
$(BIN_DIR)/graph_gen.exe: $(GRAPH_GEN_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Compile rule
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(SRC_DIR)/common.hpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Dependency on headers for BFS files
$(OBJ_DIR)/bfs_mpi.o:     $(SRC_DIR)/bfs_mpi.hpp
$(OBJ_DIR)/bfs_hybrid.o:  $(SRC_DIR)/bfs_hybrid.hpp
$(OBJ_DIR)/graph_io.o:    $(SRC_DIR)/graph_io.hpp
$(OBJ_DIR)/main.o:        $(SRC_DIR)/bfs_mpi.hpp $(SRC_DIR)/bfs_hybrid.hpp \
                          $(SRC_DIR)/graph_io.hpp

# Generate test graphs
graphs: $(BIN_DIR)/graph_gen.exe
	@echo "Generating test graphs..."
	$(BIN_DIR)/graph_gen.exe 20000  16 data/graph_20k.csr  42
	$(BIN_DIR)/graph_gen.exe 40000  16 data/graph_40k.csr  42
	$(BIN_DIR)/graph_gen.exe 80000  16 data/graph_80k.csr  42
	$(BIN_DIR)/graph_gen.exe 160000 16 data/graph_160k.csr 42
	@echo "Done."

clean:
	rm -rf $(BIN_DIR) $(OBJ_DIR)
