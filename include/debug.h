#ifndef DEBUG_H
#define DEBUG_H

#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <iomanip>
#include "utils.h"
#include "colors.h"

#ifdef DEBUG
#define print_gpu(name, gpu_array, size){ _print_gpu(name, gpu_array, size, __FILE__, __LINE__); }
#define print_cpu(name, cpu_array, size){ _print_cpu(name, cpu_array, size, __FILE__, __LINE__); }
#define check_nan(name, array, size){ _check_nan(name, array, size, __FILE__, __LINE__); }

template <typename T>
static void _print_gpu(const char* name, const T* gpu_array, size_t size, const char* filename, int line){
    T *cpu_array = (T*)malloc(sizeof(T)*size);
    cudaMemcpy(cpu_array, gpu_array, sizeof(T)*size, cudaMemcpyDeviceToHost);
    std::cout << COLOR_GREEN_BLACK << "[" << filename << ":" << line << "] " << COLOR_BLUE_BLACK
              << name << ": " << COLOR_NORMAL << std::endl;
    std::cout << std::setprecision(2);
    for(int i = 0; i < size; ++i )
        // std::cout << (cpu_array[i] > 0.5 ? cpu_array[i] : 0.0) << " ";
        std::cout << cpu_array[i] << " ";
    std::cout << std::endl;
    free(cpu_array);

}

template <typename T>
static void _print_cpu(const char* name, const T* cpu_array, size_t size, const char* filename, int line){
    std::cout << COLOR_GREEN_BLACK << "[" << filename << ":" << line << "] " << COLOR_BLUE_BLACK
              << name << ": " << COLOR_NORMAL << std::endl;
    for(int i = 0; i < size; ++i )
        std::cout << std::hex << cpu_array[i] << "|";
    std::cout << std::endl;
}
static void _check_nan(const char* name, const float* a, size_t size, const char* filename, int line){
    auto result = has_nan(a, size) ? "yes":"no";
    std::cout << COLOR_GREEN_BLACK << "[" << filename << ":" << std::setw(3) << line << "] " << COLOR_BLUE_BLACK << "checking NaN --> "
              << name << ": " << COLOR_NORMAL << result << std::endl;
}

#else

#define print_gpu(name, gpu_array, size)
#define print_cpu(name, gpu_array, size)
#define check_nan(name, array, size)

#endif

#endif
