#ifndef ERROR_H
#define ERROR_H

#include <iostream>
#include "colors.h"

#define throw_error(message) \
    std::cerr << COLOR_YELLOW_BLACK << "[Error] " << COLOR_BOLD << COLOR_RED_BLACK << message << COLOR_NORMAL << std::endl
#define throw_error_with_reason(message) \
    std::cerr << COLOR_YELLOW_BLACK << "[Error] " << COLOR_BOLD << COLOR_RED_BLACK << message << "  " << COLOR_RED_WHITE << (errno ? strerror(errno) : "") << COLOR_NORMAL << std::endl

#endif
