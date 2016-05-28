#ifndef MESSAGES_H
#define MESSAGES_H

#include <iostream>
#include <iomanip>
#include "colors.h"

#define throw_error(message) \
    std::cerr << COLOR_YELLOW_BLACK << "[Error] " << COLOR_BOLD << COLOR_RED_BLACK << message << COLOR_NORMAL << std::endl
#define throw_error_with_reason(message) \
    std::cerr << COLOR_YELLOW_BLACK << "[Error] " << COLOR_BOLD << COLOR_RED_BLACK << message << "  " << COLOR_RED_WHITE << (errno ? strerror(errno) : "") << COLOR_NORMAL << std::endl

#define print_train_error(epoch, cost) \
    std::cout << std::setprecision(3)<< COLOR_BOLD_GREEN_BLACK << "[epoch = " << std::setw(3) << epoch << "]  " << COLOR_BLUE_BLACK << "error: " << cost << COLOR_NORMAL << std::endl

#endif
