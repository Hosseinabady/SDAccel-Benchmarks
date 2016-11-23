#include "vector_addition.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <math.h>

//OpenCL includes
#include "xcl.h"



int main(int argc, char* argv[])
{

    if(argc != 2)
    {
        std::cout << "Usage: " << argv[0] <<" <xclbin>" << std::endl;
        return -1;
    }

    const char* xclbinFilename = argv[1];


    xcl_world world;
    cl_kernel krnl;

    if(strstr(argv[1], ".xclbin") != NULL) {
        world = xcl_world_single(CL_DEVICE_TYPE_ACCELERATOR);
        krnl  = xcl_import_binary(world, xclbinFilename, "vector_addition");
    } else {
        world = xcl_world_single(CL_DEVICE_TYPE_CPU);
        krnl  = xcl_import_source(world, xclbinFilename, "vector_addition");
    }


    size_t vector_size_bytes = sizeof(float) * DATA_LENGTH;
    cl_mem buffer_a = xcl_malloc(world, CL_MEM_READ_ONLY, vector_size_bytes);
    cl_mem buffer_b = xcl_malloc(world, CL_MEM_READ_ONLY, vector_size_bytes);
    cl_mem buffer_c = xcl_malloc(world, CL_MEM_READ_ONLY, vector_size_bytes);

    float *source_a = (float *) malloc(vector_size_bytes);
    float *source_b = (float *) malloc(vector_size_bytes);
    float *source_c = (float *) malloc(vector_size_bytes);


    /* Create the test data and run the vector addition locally */
    for(int i = 0; i < DATA_LENGTH; i++) {
    	source_a[i] = 2.4;
    	source_b[i] = 23.2;
    	source_c[i] = 0.1;
    }

    /* Copy input vectors to memory */
    xcl_memcpy_to_device(world,buffer_a,source_a,vector_size_bytes);
    xcl_memcpy_to_device(world,buffer_b,source_b,vector_size_bytes);

    /* Release the memory for temporary source data buffers on the host */
    //free(source_a);

    /* Set the kernel arguments */
    int vector_length = DATA_LENGTH;
    clSetKernelArg(krnl, 0, sizeof(cl_mem), &buffer_a);
    clSetKernelArg(krnl, 1, sizeof(cl_mem), &buffer_b);
    clSetKernelArg(krnl, 2, sizeof(cl_mem), &buffer_c);

    /* Launch the kernel */
    unsigned long duration = xcl_run_kernel3d(world, krnl, DATA_LENGTH , 1, 1);


    printf("duration = %lu\n", duration);

    xcl_memcpy_from_device(world, source_c,  buffer_c, vector_size_bytes);

    for (int i = 0; i < DATA_LENGTH; i++) {
    	float gold_result = source_a[i] + source_b[i];
    	float diff = fabs(gold_result-source_c[i]);
    	if (diff > 0.001) {
			printf("Validation Error; arrayRef[%d] = %f != arrayC[%d] = %f\n", i, gold_result, i, source_c[i]);
			exit(1);
    	}
    }

    printf("Result OK\n");



    free(source_a);
    free(source_b);
    free(source_c);
    clReleaseMemObject(buffer_a);
    clReleaseMemObject(buffer_b);
    clReleaseMemObject(buffer_c);
    clReleaseKernel(krnl);

    xcl_release_world(world);


}
