//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains functions to read the MNIST dataset (less features, Visual Studio friendly)
 *
 * This header should only be used with old compilers.
 */

#ifndef MNIST_READER_HPP
#define MNIST_READER_HPP

#include <fstream>
#include <iostream>
#include <vector>
#include <cstdint>
#include <memory>

#include "mnist_reader_helper.hpp"

namespace mnist {

/*!
 * \brief Read a MNIST image file and return a container filled with the images
 * \param path The path to the image file
 * \return A std::vector filled with the read images
 */
void read_mnist_info(const std::string& path, int& num_imgs, int& size_img, unsigned char* buffer) {
    auto buffer = read_mnist_file(path, 0x803);

    if (buffer) {
        num_imgs = read_header(buffer, 1);
        // size of each img = rows * columns
        size = read_header(buffer, 2) * read_header(buffer, 3);
        // Skip the header
        // Cast to unsigned char is necessary cause signedness of char is platform-specific
        buffer = reinterpret_cast<unsigned char*>(buffer.get() + 16);
    }
    else {
        num_imgs = 0;
        size     = 0;
        buffer   = nullptr;
    }
}

/*!
 * \brief Read a MNIST label file and return a container filled with the labels
 * \param path The path to the image file
 * \return A std::vector filled with the read labels
 */
void read_mnist_label_file(const std::string& path) {
    auto buffer = read_mnist_file(path, 0x801);

    if (buffer) {
        auto count = read_header(buffer, 1);

        //Skip the header
        //Cast to unsigned char is necessary cause signedness of char is
        //platform-specific
        // auto label_buffer = reinterpret_cast<unsigned char*>(buffer.get() + 8);

        // std::vector<Label> labels(count);

        // for (size_t i = 0; i < count; ++i) {
            // auto label = *label_buffer++;
            // labels[i]  = static_cast<Label>(label);
        // }

        return labels;
    }
}

} //end of namespace mnist

#endif

