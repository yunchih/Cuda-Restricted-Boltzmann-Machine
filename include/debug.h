#ifndef DEBUG_H
#define DEBUG_H

#include <iostream>

#ifdef DEBUG
#define print_gpu(name, gpu_array, size){ _print_gpu(name, gpu_array, size, __FILE__, __LINE__); }
#define print_cpu(name, cpu_array, size){ _print_cpu(name, cpu_array, size, __FILE__, __LINE__); }

template <typename T>
static void _print_gpu(const char* name, const T* gpu_array, size_t size, const char* filename, int line){
    T *cpu_array = (T*)malloc(sizeof(T)*size);
    cudaMemcpy(cpu_array, gpu_array, sizeof(T)*size, cudaMemcpyDeviceToHost);
    std::cout << "[" << filename << ":" << line << "] "
              << name << ": " << std::endl;
    for(int i = 0; i < size; ++i )
        std::cout << cpu_array[i] << " ";
    std::cout << std::endl;
    free(cpu_array);

}

template <typename T>
static void _print_cpu(const char* name, const T* cpu_array, size_t size, const char* filename, int line){
    std::cout << "[" << filename << ":" << line << "] "
              << name << ": " << std::endl;
    for(int i = 0; i < size; ++i )
        std::cout << std::hex << cpu_array[i] << "|";
    std::cout << std::endl;
}

#else

#define print_gpu(name, gpu_array, size, type)
#define print_cpu(name, gpu_array, size, type)

#endif

#endif
