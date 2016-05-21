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

class MnistReader {

private:
    char* buffer;

    uint32_t read_header(size_t position) {
        auto header = reinterpret_cast<uint32_t*>(this->buffer);
        auto value = *(header + position);
        return (value << 24) | ((value << 8) & 0x00FF0000) | ((value >> 8) & 0X0000FF00) | (value >> 24);
    }

    void read_mnist_file(const std::string& path, uint32_t key) {
        std::ifstream file;
        file.open(path, std::ios::in | std::ios::binary | std::ios::ate);

        if (!file) {
            std::cout << "Error opening file" << std::endl;
            exit(1);
        }

        auto size = file.tellg();
        this->buffer = new char(size);

        //Read the entire file at once
        file.seekg(0, std::ios::beg);
        file.read(this->buffer, size);
        file.close();

        auto magic = read_header(0);

        if (magic != key) {
            std::cout << "Invalid magic number, probably not a MNIST file" << std::endl;
            exit(1);
        }

        auto count = read_header(1);

        if (magic == 0x803) {
            auto rows    = read_header(2);
            auto columns = read_header(3);

            if (size < count * rows * columns + 16) {
                std::cout << "The file is not large enough to hold all the data, probably corrupted" << std::endl;
                exit(1);
            }
        } else if (magic == 0x801) {
            if (size < count + 8) {
                std::cout << "The file is not large enough to hold all the data, probably corrupted" << std::endl;
                exit(1);
            }
        }
    }

public:
    ~MnistReader(){

    }
    void read_mnist_info(const std::string& path, int& num_imgs, int& size_img) {
        read_mnist_file(path, 0x803);

        if (this->buffer) {
            num_imgs = read_header(1);
            // size of each img = rows * columns
            size_img = read_header(2) * read_header(3);
            // Skip the header
            this->buffer += 16;
        }
        else {
            num_imgs     = 0;
            size_img     = 0;
            this->buffer = nullptr;
        }
    }

    const unsigned char* read_img_minibatch(int minibatch_index, int minibatch_size){
        return (unsigned char*)buffer + minibatch_index*minibatch_size;
    };
};

#endif

