cmake_minimum_required(VERSION 3.16)
project(HybridBFS VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Common compile options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")

# ---- BFS Executable ----
add_executable(bfs
    src/main.cpp
    src/bfs_mpi.cpp
    src/bfs_hybrid.cpp
    src/graph_io.cpp
)
target_include_directories(bfs PRIVATE src)
target_link_libraries(bfs PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)

# ---- Graph Generator Executable ----
add_executable(graph_gen
    src/graph_generator.cpp
    src/graph_io.cpp
)
target_include_directories(graph_gen PRIVATE src)
target_link_libraries(graph_gen PRIVATE OpenMP::OpenMP_CXX)
