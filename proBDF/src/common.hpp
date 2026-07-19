#ifndef COMMON_HPP
#define COMMON_HPP

#include <cstdint>
#include <vector>
#include <chrono>
#include <string>
#include <algorithm>
#include <numeric>
#include <cstring>

// ============================================================
// CSR Graph representation (Compressed Sparse Row)
// ============================================================
struct CsrGraph {
    int64_t V;        // total vertex count (global)
    int64_t E;        // total edge count (global)
    int64_t local_V;  // vertices owned by this rank
    int64_t offset;   // first global vertex index owned by this rank

    // Full CSR (only meaningful on rank 0 before distribution, or small graphs)
    std::vector<int64_t> row_ptr;   // size V+1, offsets into col_idx
    std::vector<int>     col_idx;   // size E, destination vertices

    // Local CSR view (each rank after distribution)
    std::vector<int64_t> local_row_ptr;  // size local_V+1
    std::vector<int>     local_col_idx;  // edges from local vertices

    CsrGraph() : V(0), E(0), local_V(0), offset(0) {}

    /// Convert local vertex index (0..local_V-1) to global vertex ID
    inline int local_to_global(int64_t local_v) const {
        return static_cast<int>(offset + local_v);
    }

    /// Determine which rank owns a global vertex, given P total ranks
    inline int owner_rank(int global_v, int P) const {
        int64_t per_rank = (V + P - 1) / P;
        int r = global_v / static_cast<int>(per_rank);
        return (r < P) ? r : (P - 1);
    }

    /// Convert a global vertex ID to a local index for a given rank
    inline int64_t global_to_local(int global_v, int P) const {
        int64_t per_rank = (V + P - 1) / P;
        int r = global_v / static_cast<int>(per_rank);
        if (r >= P) r = P - 1;
        return global_v - r * per_rank;
    }

    /// Get starting offset for rank r
    inline int64_t rank_offset(int r, int P) const {
        int64_t per_rank = (V + P - 1) / P;
        return r * per_rank;
    }

    /// Get number of vertices owned by rank r
    inline int64_t rank_count(int r, int P) const {
        int64_t per_rank = (V + P - 1) / P;
        int64_t start = r * per_rank;
        int64_t end = std::min(start + per_rank, V);
        return end - start;
    }
};

// ============================================================
// BFS Statistics
// ============================================================
struct BfsStats {
    double   time_ms;           // wall-clock time for BFS traversal
    int      levels;            // number of BFS levels
    int64_t  visited_count;     // total vertices visited (global sum)
    std::vector<int64_t> frontier_sizes;  // size of frontier at each level (local)

    BfsStats() : time_ms(0.0), levels(0), visited_count(0) {}
};

// ============================================================
// High-resolution timer
// ============================================================
class Timer {
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;
    TimePoint start_;
public:
    Timer() : start_(Clock::now()) {}
    void reset() { start_ = Clock::now(); }
    double elapsed_ms() const {
        auto end = Clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }
    double elapsed_us() const {
        auto end = Clock::now();
        return std::chrono::duration<double, std::micro>(end - start_).count();
    }
};

#endif // COMMON_HPP
