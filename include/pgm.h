#ifdef PGM_WRITER_H
#define PGM_WRITER_H
#include <err.h>
#include <cstdio>
#include <memory>
#include <cstdint>

class PGM_Writer {

private:
    FILE* out;
public:
    PGM_Writer(const char* out_file, int width, int height){
        const int num_channel = 1;
        const int num_pixel   = width*height;

        FILE *out = fopen(out_file, "wb");
        if(out == NULL)
            errx("Fail opening %s\n", out_file);
        fprintf(out, "P5\n%d %d\n255\n", width, height);
    }
    void write(uint8_t* i, int size){
        fwrite(i, 1, size, out);
    }
    void done(){
        fclose(out);
    }
};
#endif
