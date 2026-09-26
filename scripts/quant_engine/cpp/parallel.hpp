// Minimal task-parallel loop built on std::thread (no OpenMP dependency, so
// the same code compiles with GCC, Clang and MSVC).
#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace qe {

inline int resolve_threads(int n_threads) {
    if (n_threads > 0) return n_threads;
    const unsigned hw = std::thread::hardware_concurrency();
    return hw == 0 ? 1 : static_cast<int>(hw);
}

// Calls fn(i) for i = 0, ..., n_tasks - 1. Tasks are handed out dynamically;
// callers must write results to task-specific slots so that the outcome does
// not depend on the scheduling order.
template <class Fn>
void parallel_for(std::size_t n_tasks, int n_threads, Fn&& fn) {
    const std::size_t workers =
        std::min<std::size_t>(static_cast<std::size_t>(resolve_threads(n_threads)), n_tasks);
    if (workers <= 1) {
        for (std::size_t i = 0; i < n_tasks; ++i) fn(i);
        return;
    }
    std::atomic<std::size_t> next{0};
    std::exception_ptr error;
    std::mutex error_mutex;
    auto worker = [&]() {
        for (;;) {
            const std::size_t i = next.fetch_add(1);
            if (i >= n_tasks) return;
            try {
                fn(i);
            } catch (...) {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!error) error = std::current_exception();
            }
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(workers);
    for (std::size_t w = 0; w < workers; ++w) pool.emplace_back(worker);
    for (auto& thread : pool) thread.join();
    if (error) std::rethrow_exception(error);
}

}  // namespace qe
