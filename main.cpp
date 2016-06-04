#include "rbm.h"
#include "colors.h"
#include "messages.h"
#include <iostream>
#include <iomanip>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
int change_dir(const char* dir_name){
    struct stat st = {0};
    /* Create directory if not exists */
    if (stat(dir_name, &st) == -1) {
        if(mkdir(dir_name, 0777) == -1){
            throw_error_with_reason("Fail creating directory: " << dir_name);
            exit(1);
        }
    }
    if(chdir(dir_name) == -1){
        throw_error_with_reason("Fail changing directory to: " << dir_name);
        exit(1);
    }
}
int main(int argc, char *argv[])
{
    if(argc != 10){
        std::cerr << COLOR_BOLD_RED_BLACK << "Usage: "
                  << COLOR_BOLD_BLUE_BLACK
                  << argv[0] 
                  << " [Output directory] [Training data filename]"
                  << " [Test data filename] [Learning rate]"
                  << " [Epoch number] [CD step] [Train data size]"
                  << " [Test data size] [Random sample size]"
                  << COLOR_NORMAL
                  << std::endl;
        exit(1);
    }
    const char* out_dir         = argv[1];
    const char* train_data_file = argv[2];
    const char* test_data_file  = argv[3];
    const float learning_rate   = atof(argv[4]);
    const int   n_epoch         = atoi(argv[5]);
    const int   n_cd            = atoi(argv[6]);
    const int   train_size      = atoi(argv[7]);
    const int   test_size       = atoi(argv[8]);
    const int   sample_size     = atoi(argv[9]);

    std::cout << COLOR_BOLD << "Starting RBM with the following configurations:" << COLOR_NORMAL << std::endl;
    std::cout << COLOR_GREEN_BLACK;
    std::cout << std::setw(20) << "[Output directory]"   << " = " << out_dir         << std::endl;
    std::cout << std::setw(20) << "[Training data]"      << " = " << train_data_file << std::endl;
    std::cout << std::setw(20) << "[Validation data]"    << " = " << test_data_file  << std::endl;
    std::cout << std::setw(20) << "[Learning rate]"      << " = " << learning_rate   << std::endl;
    std::cout << std::setw(20) << "[Epoch number]"       << " = " << n_epoch         << std::endl;
    std::cout << std::setw(20) << "[CD step]"            << " = " << n_cd            << std::endl;
    std::cout << std::setw(20) << "[Train data size]"    << " = " << train_size      << std::endl;
    std::cout << std::setw(20) << "[Test data size]"     << " = " << test_size       << std::endl;
    std::cout << std::setw(20) << "[Random sample size]" << " = " << sample_size     << std::endl;
    std::cout << COLOR_NORMAL << std::endl;

    MnistReader train_data_reader(train_data_file, train_size);
    MnistReader test_data_reader(test_data_file, test_size);
    change_dir(out_dir);

    RBM rbm(784, 512, learning_rate, n_epoch, n_cd, sample_size, train_data_reader, test_data_reader, std::make_pair(28,28));
    rbm.train();
    return 0;
}
