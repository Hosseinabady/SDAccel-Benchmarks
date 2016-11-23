
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <CL/opencl.h>

#include "vector_addition.h"

//#define GLOBAL_SIZE(x)	ceil(n/(float)x)*x;
#define GLOBAL_SIZE(x)	1;

int load_file_to_memory(const char *filename, char **result);
void vector_addition_golden(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, int n);




int main(int argc, char** argv) {

	printf("From main: Hello Vector addition\n");
	printf("From main: =====================\n");

	int n = DATA_LENGTH;


	DATA_TYPE *h_A;
	DATA_TYPE *h_B;


	DATA_TYPE *h_C;
	DATA_TYPE *h_C_golden;


	int err;
    size_t global[1];                   // global domain size
    size_t local[1];                    // local domain size

    cl_platform_id platform_id;         // platform id
    cl_device_id device_id;             // compute device id
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;      // compute programs

    cl_kernel read_kernel;                   // compute mean kernel
    cl_kernel addition_kernel;                   // compute reduce kernel
    cl_kernel write_kernel;

    char cl_platform_vendor[1001];
    char cl_platform_name[1001];

    cl_mem d_A;                         // device memory used for data
    cl_mem d_B;                         // device memory used for mean
    cl_mem d_C;


    cl_ulong time_start, time_end;
    double total_time;


    h_A = (DATA_TYPE*)malloc(sizeof(DATA_TYPE)*n);
    h_B = (DATA_TYPE*)malloc(sizeof(DATA_TYPE)*n);

    h_C = (DATA_TYPE*)malloc(sizeof(DATA_TYPE)*n);
    h_C_golden = (DATA_TYPE*)malloc(sizeof(DATA_TYPE)*n);


    //initialization
    DATA_TYPE t;
    for(int i = 0; i < n; i++) {

    	DATA_TYPE t;

    	t = rand()/(RAND_MAX*1.0);
    	h_A[i] = t;

    	t = rand()/(RAND_MAX*1.0);
    	h_B[i] = t;

    }

    for(int i = 0; i < n; i++) {
   		h_C[i] = 0;
   		h_C_golden[i] = 0;
    }

	 // Connect to first platform
	 //
	 err = clGetPlatformIDs(1,&platform_id,NULL);
	 if (err != CL_SUCCESS) {
		 printf("Error: Failed to find an OpenCL platform!\n");
	     printf("Test failed\n");
	     return EXIT_FAILURE;
	 }
		{
			int num_platforms = 1;
			char buffer[10240];
			printf(" %d platform(s) found\n", num_platforms);
			printf(" =====================\n");
			printf("\n");

			for (int i = 0; i <num_platforms; i++) {
				printf("platform number %d \n", i);
				printf("------------------------\n");
				clGetPlatformInfo(platform_id, CL_PLATFORM_PROFILE, 10240, buffer, NULL);
				printf("  CL_PLATFORM_PROFILE = %s\n", buffer);

				clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION, 10240, buffer, NULL);
				printf("  CL_PLATFORM_VERSION = %s\n", buffer);

				clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, 10240, buffer, NULL);
				printf("  CL_PLATFORM_NAME = %s\n", buffer);

				clGetPlatformInfo(platform_id, CL_PLATFORM_VENDOR, 10240, buffer, NULL);
				printf("  CL_PLATFORM_VENDOR = %s\n", buffer);

				clGetPlatformInfo(platform_id, CL_PLATFORM_EXTENSIONS, 10240, buffer, NULL);
				printf("  CL_PLATFORM_EXTENSIONS = %s\n", buffer);



			}
			printf("\n");
			printf("\n");
			printf("\n");
		}


	  // Connect to a compute device
	    //
	err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ACCELERATOR,
	                         1, &device_id, NULL);
	if (err != CL_SUCCESS) {
		printf("Error: Failed to create a device group!\n");
	    printf("Test failed\n");
	    return EXIT_FAILURE;
	}


	{
		char     buffer[10240];
		cl_ulong buf_ulong;
		cl_uint  buf_uint;
		size_t   buf_size_arr[3];
		size_t   buf_size;

		printf(" 1 device found\n");
		printf(" =====================\n");
		printf("\n");
		printf("------------------------\n");
		clGetDeviceInfo(device_id, CL_DEVICE_NAME, 10240, buffer, NULL);
		printf("  CL_DEVICE_NAME = %s\n", buffer);

		clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, 10240, buffer, NULL);
		printf("  CL_DEVICE_VENDOR = %s\n", buffer);

		clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(buf_uint), &buf_uint, NULL);
		printf("  CL_DEVICE_MAX_COMPUTE_UNITS = %u\n",  buf_uint);

		clGetDeviceInfo(device_id, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(buf_uint), &buf_uint, NULL);
		printf("  CL_DEVICE_MAX_CLOCK_FREQUENCY = %u\n",  buf_uint);

		clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(buf_ulong), &buf_ulong, NULL);
		printf("  CL_DEVICE_GLOBAL_MEM_SIZE = %lu\n",  buf_ulong);

		clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(buf_ulong), &buf_ulong, NULL);
		printf("  CL_DEVICE_LOCAL_MEM_SIZE = %lu\n",  buf_ulong);

		clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES , sizeof(buf_size_arr), buf_size_arr, NULL);
		printf("  CL_DEVICE_MAX_WORK_ITEM_SIZES = %lu/%lu/%lu \n", buf_size_arr[0], buf_size_arr[1], buf_size_arr[2]);

		clGetDeviceInfo(device_id,  CL_DEVICE_MAX_WORK_GROUP_SIZE , sizeof(buf_size), &buf_size, NULL);
		printf("  CL_DEVICE_MAX_WORK_GROUP_SIZE = %lu \n", buf_size);

		printf("\n");
		printf("\n");
		printf("\n");
	}
	// Create a compute context
	//
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if (!context) {
		printf("Error: Failed to create a compute context!\n");
	    printf("Test failed\n");
	    return EXIT_FAILURE;
	}

	// Create a command commands
	//
	commands = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE|CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
	if (!commands) {
		printf("Error: Failed to create a command commands!\n");
	    printf("Error: code %i\n",err);
	    printf("Test failed\n");
	    return EXIT_FAILURE;
	}

	int status;


	// Load binary from disk

	unsigned char* kernelbinary;
	char *xclbin = argv[1];


	//------------------------------------------------------------------------------
	// xclbin mean
	//------------------------------------------------------------------------------
	printf("INFO: loading xclbin_mean %s\n", xclbin);
	int n_i = load_file_to_memory(xclbin, (char **) &kernelbinary);
	if (n_i < 0) {
		printf("failed to load kernel from xclbin_mean: %s\n", xclbin);
	    printf("Test failed\n");
	    return EXIT_FAILURE;
	}

	size_t n0 = n_i;

	// Create the compute program from offline
	program = clCreateProgramWithBinary(context, 1, &device_id, &n0,
	                                        (const unsigned char **) &kernelbinary, &status, &err);

	if ((!program) || (err!=CL_SUCCESS)) {
		printf("Error: Failed to create compute program0 from binary %d!\n", err);
	    printf("Test failed\n");
	    return EXIT_FAILURE;
	}

	// Build the program executable
	//
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS) {
		size_t len;
	    char buffer[2048];

	    printf("Error: Failed to build program executable!\n");
	    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
	    printf("%s\n", buffer);
	    printf("Test failed\n");
	    return EXIT_FAILURE;
	}

	// Create the compute kernel in the program we wish to run
	//
	read_kernel = clCreateKernel(program, "read_data_kernel", &err);
	if (!read_kernel || err != CL_SUCCESS) {
		printf("Error: Failed to create read_data_kernel!\n");
	    printf("Test failed\n");
	    return EXIT_FAILURE;
	}

	addition_kernel = clCreateKernel(program, "add_data_kernel", &err);
	if (!addition_kernel || err != CL_SUCCESS) {
		printf("Error: Failed to create addition_data_kernel!\n");
	    printf("Test failed\n");
	    return EXIT_FAILURE;
	}


	write_kernel = clCreateKernel(program, "write_data_kernel", &err);
	if (!write_kernel || err != CL_SUCCESS) {
		printf("Error: Failed to create write_data_kernel !\n");
	    printf("Test failed\n");
	    return EXIT_FAILURE;
	}

    //------------------------------------------------------------------------------





    // Create the input and output arrays in device memory for our calculation
	//
	d_A = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(DATA_TYPE) * n, NULL, NULL);
	d_B = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(DATA_TYPE) * n, NULL, NULL);
	d_C = clCreateBuffer(context,  CL_MEM_WRITE_ONLY, sizeof(DATA_TYPE) * n, NULL, NULL);

	if (!d_A || !d_B || !d_C) {
		printf("Error: Failed to allocate device memory!\n");
	    printf("Test failed\n");
	    return EXIT_FAILURE;
	}

	// Write our data set into the input array in device memory
	//
	cl_event transfer_a_event;
	err = clEnqueueWriteBuffer(commands, d_A, CL_TRUE, 0, sizeof(DATA_TYPE) * n, h_A, 0, NULL, &transfer_a_event);
	if (err != CL_SUCCESS) {
		printf("Error: Failed to write to source array a!\n");
	    printf("Test failed\n");
	    return EXIT_FAILURE;
	}

	cl_event transfer_b_event;
	err = clEnqueueWriteBuffer(commands, d_B, CL_TRUE, 0, sizeof(DATA_TYPE) * n, h_B, 0, NULL, &transfer_b_event);
	if (err != CL_SUCCESS) {
		printf("Error: Failed to write to source array a!\n");
	    printf("Test failed\n");
	    return EXIT_FAILURE;
	}

	// Set the arguments to our mean kernel
	//
	err = 0;
	err  = clSetKernelArg(read_kernel, 0, sizeof(cl_mem), &d_A);
	err |= clSetKernelArg(read_kernel, 1, sizeof(cl_mem), &d_B);
	if (err != CL_SUCCESS) {
		printf("Error: Failed to set kernel arguments! %d\n", err);
	    printf("Test failed\n");
	    return EXIT_FAILURE;
	}


	cl_event read_kernel_event;
	local[0]  = WORK_GROUP_SIZE;
	global[0] = GLOBAL_SIZE(DATA_LENGTH);
	err = clEnqueueNDRangeKernel(commands, read_kernel, 1, NULL,
	                                   (size_t*)&global, (size_t*)&local, 0, NULL, &read_kernel_event);
	if (err) {
		printf("Error: Failed to execute kernel! %d\n", err);
	    printf("Test failed\n");
	    return EXIT_FAILURE;
	}



	int cnt=0;
	// Set the arguments to our reduce kernel
	//
	err = 0;
	err  = clSetKernelArg(addition_kernel, 0, sizeof(int), &cnt);

    if (err != CL_SUCCESS) {
    	printf("Error: Failed to set reduce kernel arguments! %d\n", err);
		printf("Test failed\n");
		return EXIT_FAILURE;
	}

    cl_event add_kernel_event;
	local[0]  = WORK_GROUP_SIZE;
	global[0] = GLOBAL_SIZE(DATA_LENGTH);
	err = clEnqueueNDRangeKernel(commands, addition_kernel, 1, NULL,
		                                   (size_t*)&global, (size_t*)&local, 0, NULL, &add_kernel_event);

	if (err) {
		printf("Error: Failed to execute kernel! %d\n", err);
		printf("Test failed\n");
		return EXIT_FAILURE;
	}


	err = 0;
	err  = clSetKernelArg(write_kernel, 0, sizeof(cl_mem), &d_C);

	if (err != CL_SUCCESS) {
		printf("Error: Failed to set reduce kernel arguments! %d\n", err);
		printf("Test failed\n");
		return EXIT_FAILURE;
	}
	cl_event write_kernel_event;
	local[0]  = WORK_GROUP_SIZE;
	global[0] = GLOBAL_SIZE(DATA_LENGTH);
	err = clEnqueueNDRangeKernel(commands, write_kernel, 1, NULL,
		                                   (size_t*)&global, (size_t*)&local, 0, NULL, &write_kernel_event);

	if (err) {
		printf("Error: Failed to execute kernel! %d\n", err);
		printf("Test failed\n");
		return EXIT_FAILURE;
	}


	// Read back the results from the device to verify the output
	//

//   	clFlush(commands);
   	clFinish(commands);

	cl_event transfer_c_event;
	err = clEnqueueReadBuffer( commands, d_C, CL_TRUE, 0, sizeof(DATA_TYPE) * n, h_C, 0, NULL, &transfer_c_event);
	if (err != CL_SUCCESS) {
		printf("Error: Failed to read output array! %d\n", err);
	    printf("Test failed\n");
	    return EXIT_FAILURE;
	}
   	clFinish(commands);



	clGetEventProfilingInfo(read_kernel_event, CL_PROFILING_COMMAND_START,
		sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(read_kernel_event, CL_PROFILING_COMMAND_END,
		sizeof(time_end), &time_end, NULL);
	total_time = time_end - time_start;
	printf("\nExecution time for read kernel in milliseconds = %0.3f ms\n", (total_time / 1000000.0));


	clGetEventProfilingInfo(add_kernel_event, CL_PROFILING_COMMAND_START,
		sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(add_kernel_event, CL_PROFILING_COMMAND_END,
		sizeof(time_end), &time_end, NULL);
	total_time = time_end - time_start;
	printf("\nExecution time for add kernel in milliseconds = %0.3f ms\n", (total_time / 1000000.0));


	clGetEventProfilingInfo(write_kernel_event, CL_PROFILING_COMMAND_START,
		sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(write_kernel_event, CL_PROFILING_COMMAND_END,
		sizeof(time_end), &time_end, NULL);
	total_time = time_end - time_start;
	printf("\nExecution time for write kernel in milliseconds = %0.3f ms\n", (total_time / 1000000.0));


	clGetEventProfilingInfo(transfer_a_event, CL_PROFILING_COMMAND_START,
		sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(transfer_a_event, CL_PROFILING_COMMAND_END,
		sizeof(time_end), &time_end, NULL);
	total_time = time_end - time_start;
	printf("\nExecution time for transfer a in milliseconds = %0.3f ms\n", (total_time / 1000000.0));

	clGetEventProfilingInfo(transfer_b_event, CL_PROFILING_COMMAND_START,
		sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(transfer_b_event, CL_PROFILING_COMMAND_END,
		sizeof(time_end), &time_end, NULL);
	total_time = time_end - time_start;
	printf("\nExecution time for transfer b in milliseconds = %0.3f ms\n", (total_time / 1000000.0));


	clGetEventProfilingInfo(transfer_c_event, CL_PROFILING_COMMAND_START,
		sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(transfer_c_event, CL_PROFILING_COMMAND_END,
		sizeof(time_end), &time_end, NULL);
	total_time = time_end - time_start;
	printf("\nExecution time for transfer c in milliseconds = %0.3f ms\n", (total_time / 1000000.0));


    vector_addition_golden(h_A, h_B, h_C_golden, n);

    for (int i = 0; i < n; i++) {
    	DATA_TYPE gold=h_C_golden[i];
	   	DATA_TYPE hw=h_C[i];
	    DATA_TYPE diff = fabs(gold-hw);
	    if (diff > 0.1) {
	    	printf("Error at element %d golden= %f, hw=%f\n", i, gold, hw);
	    	break;
	    }
	}

    printf("From main: Bye Vector addition\n");
    printf("From main: ====================\n");
    return 0;
}

void vector_addition_golden(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, int n) {

	for(int j = 0; j < n; j++) {
		C[j] = A[j] + B[j];
	}
}


