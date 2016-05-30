#ifndef PGM_WRITER_H
#define PGM_WRITER_H
#include <cstdio>
#include <memory>
#include <cstdint>
#include <string>
#include "messages.h"

class PGM_Writer {

private:
    FILE* out;
public:
    PGM_Writer(const char* out_file, int width, int height){
        out = fopen(out_file, "wb");
        if(out == NULL){
            throw_error("Fail opening " << out_file);
            exit(1);
        }
        fprintf(out, "P5\n%d %d\n255\n", width, height);
    }
    void write(uint8_t* i, int size){
        fwrite(i, 1, size, out);
    }
    ~PGM_Writer(){
        fclose(out);
    }
};
#endif
