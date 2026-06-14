#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace md::utils {
    inline void cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("CUDA error at ") + file + ":" + std::to_string(line) +
                " in " + expr + ": " + cudaGetErrorString(err)
            );
        }
    }
}

#define MD_CUDA_CHECK(expr) \
    ::md::utils::cuda_check((expr), #expr, __FILE__, __LINE__)

#define MD_CUDA_KERNEL_CHECK() \
    ::md::utils::cuda_check(cudaGetLastError(), "kernel launch", __FILE__, __LINE__)
