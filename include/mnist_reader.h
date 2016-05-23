//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef MNIST_READER_HPP
#define MNIST_READER_HPP

#include <fstream>
#include <iostream>
#include <vector>
#include <cstdint>
#include <memory>
#include "utils.h"

class MnistReader {

private:
    char* cpu_buffer;
    float* gpu_buffer;
    char* gpu_tmp;
    int each_size;
    int total_num;

    uint32_t read_header_field(size_t position) {
        auto header = reinterpret_cast<uint32_t*>(this->cpu_buffer);
        auto value = *(header + position);
        return (value << 24) | ((value << 8) & 0x00FF0000) | ((value >> 8) & 0X0000FF00) | (value >> 24);
    }

    void read_mnist_meta(const char* in_file) {
        std::ifstream file;
        file.open(in_file, std::ios::in | std::ios::binary | std::ios::ate);

        if (!file) {
            std::cerr << "Error opening file" << std::endl;
            exit(1);
        }

        auto size = file.tellg();
        this->cpu_buffer = new char(size);

        //Read the entire file at once
        file.seekg(0, std::ios::beg);
        file.read(this->cpu_buffer, size);
        file.close();

        auto magic = read_header_field(0);

        if (magic != 0x801 && magic != 0x803) {
            std::cerr << "Invalid magic number, probably not a MNIST file" << std::endl;
            exit(1);
        }

        total_num = read_header_field(1);

        /* Training data */
        if (magic == 0x803) {
            /* size = row * column */
            each_size =  read_header_field(2) * read_header_field(3);

            if (size < total_num * each_size + 16) {
                std::cerr << "The file is not large enough to hold all the data, probably corrupted" << std::endl;
                exit(1);
            }
        /* Label data */
        } else if (magic == 0x801) {
            std::cerr << in_file << " looks like a label dataset, which is not supported now" << std::endl;
            exit(1);
        }

        /* Skip the header and proceed to next memory position */
        this->cpu_buffer += 16;
    }

public:
    ~MnistReader(){
        cudaFree((void*)gpu_buffer);
        cudaFree((void*)gpu_tmp);
        delete [] this->cpu_buffer;
    }

    int get_total(){
        return total_num;
    }
    MnistReader(const char* _file){
        read_mnist_meta(_file);
        // allocate single train data
        cudaMalloc((void**)gpu_buffer, sizeof(float)*each_size);
        // allocate buffer for cpu -> gpu transformation
        cudaMalloc((void**)gpu_tmp, sizeof(char)*each_size);
    }
    const float* get_example_at(int pos){
        char* p = this->cpu_buffer + each_size * pos;
        transform_example(this->gpu_buffer, this->gpu_tmp, p, each_size);
        return this->gpu_buffer;
    }
};

#endif

