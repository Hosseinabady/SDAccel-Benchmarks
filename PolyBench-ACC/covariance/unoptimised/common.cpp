

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <CL/opencl.h>

#include "covariance.h"
int
load_file_to_memory(const char *filename, char **result)
{
    int size = 0;
    FILE *f = fopen(filename, "rb");
    if (f == NULL)
        {
            *result = NULL;
            return -1;
        }
    fseek(f, 0, SEEK_END);
    size = ftell(f);
    fseek(f, 0, SEEK_SET);
    *result = (char *)malloc(size+1);
    if (size != fread(*result, sizeof(char), size, f))
        {
            free(*result);
            return -2; // -2 means file reading fail
        }
    fclose(f);
    (*result)[size] = 0;
    return size;
}

