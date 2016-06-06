#ifndef MNIST_READER_HPP
#define MNIST_READER_HPP

#include <fstream>
#include <iostream>
#include <vector>
#include <cstdint>
#include <memory>
#include <cstdio>
#include "utils.h"
#include "messages.h"

class MnistReader {

private:
    float* cpu_buffer;
    float* gpu_buffer;
    int each_size;
    int data_num;

    struct TransformInput{
        __host__ __device__ float operator()(const uint8_t in) const {
            return (float)in / 255.0f;
        }
    };

    uint32_t read_header_field(char* header, size_t position) {
        auto _header = reinterpret_cast<uint32_t*>(header);
        auto value = *(_header + position);
        return (value << 24) | ((value << 8) & 0x00FF0000) | ((value >> 8) & 0X0000FF00) | (value >> 24);
    }

    void read_mnist_meta(const char* in_file) {
        std::ifstream file;
        file.open(in_file, std::ios::in | std::ios::binary | std::ios::ate);

        if (!file) {
            throw_error_with_reason("Fail opening file: " << in_file);
            exit(1);
        }

        auto size = file.tellg();
        char header[16];

        //Read the header
        file.seekg(0, std::ios::beg);
        file.read(header, 16);

        auto magic = read_header_field(header,0);

        if (magic != 0x801 && magic != 0x803) {
            throw_error("Invalid magic number, probably not a MNIST file");
            exit(1);
        }

        int avail_data_num = read_header_field(header,1);

        if(avail_data_num < data_num){
            throw_error("Request data size [" << data_num << "] exceed size of available data: " << avail_data_num);
            exit(1);
        }
        /* Training data */
        if (magic == 0x803) {
            /* size = row * column */
            this->each_size =  read_header_field(header,2) * read_header_field(header,3);

            if (size < avail_data_num * this->each_size + 16) {
                throw_error("The file is not large enough to hold all the data, probably corrupted");
                exit(1);
            }
        } 
        /* Label data */
        else if (magic == 0x801) {
            throw_error(in_file << " looks like a label dataset, which is not supported now");
            exit(1);
        }

        int total_size = data_num*each_size;
        char* char_buffer = new char[total_size];
        // read the whole speicified number of examples at once
        file.read(char_buffer, total_size);
        file.close();
        
        auto p = (uint8_t*)char_buffer;
        this->cpu_buffer = new float[total_size];
        thrust::transform(thrust::host, p, p + total_size, this->cpu_buffer, TransformInput());
        delete [] char_buffer; 
    }

public:
    ~MnistReader(){
        cudaErrCheck(cudaFree(this->gpu_buffer));
        delete [] this->cpu_buffer;
    }
    int get_total(){
        return this->data_num;
    }
    MnistReader(const char* _file, int _data_num):data_num(_data_num){
        read_mnist_meta(_file);

        // allocate space for single train data
        cudaErrCheck(cudaMalloc((void**)&(this->gpu_buffer), sizeof(float)*each_size));
    }
    const float* get_example_at(int pos){
        if(pos >= data_num){
            throw_error("accessing invalid example at " << pos);
            exit(1);
        }

        float* p = this->cpu_buffer + this->each_size * pos;
        cudaErrCheck(cudaMemcpy(this->gpu_buffer, p, sizeof(float)*this->each_size, cudaMemcpyHostToDevice));
        return this->gpu_buffer;
    }
};

#endif
