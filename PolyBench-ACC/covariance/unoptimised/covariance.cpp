/* File: covariance.cpp
 *
 Copyright (c) [2016] [Mohammad Hosseinabady (mohammad@hosseinabady.com)]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
===============================================================================
* This file has been written at University of Bristol
* for the ENPOWER project funded by EPSRC
*
* File name : covariance.cl.cpp
* author    : Mohammad hosseinabady mohammad@hosseinabady.com
* date      : 12 November 2016
* blog: https://highlevel-synthesis.com/
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <CL/opencl.h>

#include "covariance.h"


int load_file_to_memory(const char *filename, char **result);
void mean_golden(DATA_TYPE *mean, DATA_TYPE *data,DATA_TYPE *covar, DATA_TYPE float_n, int m, int n);





int main(int argc, char** argv) {

	printf("From main: Hello Covariance\n");
	printf("From main: ====================\n");

	int m = M;
	int n = N;

	DATA_TYPE float_n= N*1.0;

	DATA_TYPE *h_data;
	DATA_TYPE *h_data_golden;

	DATA_TYPE *h_mean;
	DATA_TYPE *h_mean_golden;

	DATA_TYPE *h_covar;
	DATA_TYPE *h_covar_golden;


	int err;
    size_t global_mean[1];                   // global domain size for our calculation
    size_t local_mean[1];                    // local domain size for our calculation

    size_t global_reduce[2];                   // global domain size for our calculation
    size_t local_reduce[2];                    // local domain size for our calculation


    size_t global_cover[1];                   // global domain size for our calculation
    size_t local_cover[1];                    // local domain size for our calculation
    cl_platform_id platform_id;         // platform id
    cl_device_id device_id;             // compute device id
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;      // compute programs
    cl_kernel kernel_mean;                   // compute mean kernel
    cl_kernel kernel_reduce;                   // compute reduce kernel
    cl_kernel kernel_covar;
    char cl_platform_vendor[1001];
    char cl_platform_name[1001];

    cl_mem d_data;                         // device memory used for data
    cl_mem d_mean;                         // device memory used for mean
    cl_mem d_covar;


    h_data = (DATA_TYPE*)malloc(sizeof(DATA_TYPE)*n*m);
    h_data_golden = (DATA_TYPE*)malloc(sizeof(DATA_TYPE)*n*m);

    h_mean = (DATA_TYPE*)malloc(sizeof(DATA_TYPE)*m);
    h_mean_golden = (DATA_TYPE*)malloc(sizeof(DATA_TYPE)*m);

    h_covar = (DATA_TYPE*)malloc(sizeof(DATA_TYPE)*m*m);
    h_covar_golden = (DATA_TYPE*)malloc(sizeof(DATA_TYPE)*m*m);

    //initialization
    DATA_TYPE t;
    for(int i = 0; i < m; i++) {
    	h_mean[i] = 0;
    	h_mean_golden[i] = 0;
    	for(int j = 0; j < n; j++) {
    		t = 3.2;//rand()/(RAND_MAX*1.0);
    		h_data[j*m+i] = t;
    		h_data_golden[j*m+i] = t;
    	}
    }
    for(int i = 0; i < m; i++) {
    	for(int j = 0; j < m; j++) {
    		h_covar[i*m+j] = 0;
    		h_covar_golden[i*m+j] = 0;
    	}
    }

	 // Connect to first platform
	 //
	 err = clGetPlatformIDs(1,&platform_id,NULL);
	 if (err != CL_SUCCESS) {
		 printf("Error: Failed to find an OpenCL platform!\n");
	     printf("Test failed\n");
	     return EXIT_FAILURE;
	 }
	 err = clGetPlatformInfo(platform_id,CL_PLATFORM_VENDOR,1000,(void *)cl_platform_vendor,NULL);
	 if (err != CL_SUCCESS) {
		 printf("Error: clGetPlatformInfo(CL_PLATFORM_VENDOR) failed!\n");
	     printf("Test failed\n");
	     return EXIT_FAILURE;
	 }
	 printf("INFO: CL_PLATFORM_VENDOR %s\n",cl_platform_vendor);
	 err = clGetPlatformInfo(platform_id,CL_PLATFORM_NAME,1000,(void *)cl_platform_name,NULL);
	 if (err != CL_SUCCESS) {
		 printf("Error: clGetPlatformInfo(CL_PLATFORM_NAME) failed!\n");
	     printf("Test failed\n");
	     return EXIT_FAILURE;
	 }
	 printf("INFO: CL_PLATFORM_NAME %s\n",cl_platform_name);



	  // Connect to a compute device
	    //
	    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ACCELERATOR,
	                         1, &device_id, NULL);
	    if (err != CL_SUCCESS) {
	            printf("Error: Failed to create a device group!\n");
	            printf("Test failed\n");
	            return EXIT_FAILURE;
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
	    commands = clCreateCommandQueue(context, device_id, 0, &err);
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
	    kernel_mean = clCreateKernel(program, "mean_kernel", &err);
	    if (!kernel_mean || err != CL_SUCCESS) {
	        printf("Error: Failed to create mean kernel!\n");
	        printf("Test failed\n");
	        return EXIT_FAILURE;
	    }

	    kernel_reduce = clCreateKernel(program, "reduce_kernel", &err);
	    if (!kernel_mean || err != CL_SUCCESS) {
	        printf("Error: Failed to create reduce kernel!\n");
	        printf("Test failed\n");
	        return EXIT_FAILURE;
	    }


	    kernel_covar = clCreateKernel(program, "covar_kernel", &err);
	    if (!kernel_covar || err != CL_SUCCESS) {
	        printf("Error: Failed to create covar kernel!\n");
	        printf("Test failed\n");
	        return EXIT_FAILURE;
	    }

	    //------------------------------------------------------------------------------





	    // Create the input and output arrays in device memory for our calculation
	     //
	     d_data = clCreateBuffer(context,  CL_MEM_READ_WRITE,  sizeof(DATA_TYPE) * n*m, NULL, NULL);
	     d_mean = clCreateBuffer(context,  CL_MEM_READ_WRITE,  sizeof(DATA_TYPE) * m, NULL, NULL);
	     d_covar = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(DATA_TYPE) * m*m, NULL, NULL);

	     if (!d_data || !d_mean || !d_covar) {
	         printf("Error: Failed to allocate device memory!\n");
	         printf("Test failed\n");
	         return EXIT_FAILURE;
	     }

	     // Write our data set into the input array in device memory
	     //
	     err = clEnqueueWriteBuffer(commands, d_data, CL_TRUE, 0, sizeof(DATA_TYPE) * n*m, h_data, 0, NULL, NULL);
	     if (err != CL_SUCCESS) {
	         printf("Error: Failed to write to source array a!\n");
	         printf("Test failed\n");
	         return EXIT_FAILURE;
	     }

	     // Set the arguments to our mean kernel
	      //
	      err = 0;
	      err  = clSetKernelArg(kernel_mean, 0, sizeof(cl_mem), &d_mean);
	      err |= clSetKernelArg(kernel_mean, 1, sizeof(cl_mem), &d_data);
	      err |= clSetKernelArg(kernel_mean, 2, sizeof(DATA_TYPE), (void *)&float_n);
	      err |= clSetKernelArg(kernel_mean, 3, sizeof(int), &m);
	      err |= clSetKernelArg(kernel_mean, 4, sizeof(int), &n);
	      if (err != CL_SUCCESS) {
	          printf("Error: Failed to set kernel arguments! %d\n", err);
	          printf("Test failed\n");
	          return EXIT_FAILURE;
	      }


	      cl_event readevent_mean;
	      local_mean[0]  = m;
	      global_mean[0] = ceil(m/(float)local_mean[0])*local_mean[0];
	      err = clEnqueueNDRangeKernel(commands, kernel_mean, 1, NULL,
	                                   (size_t*)&global_mean, (size_t*)&local_mean, 0, NULL, &readevent_mean);

	      if (err) {
	               printf("Error: Failed to execute kernel! %d\n", err);
	               printf("Test failed\n");
	               return EXIT_FAILURE;
	           }



	      // Set the arguments to our reduce kernel
		      //
		      err = 0;
		      err  = clSetKernelArg(kernel_reduce, 0, sizeof(cl_mem), &d_mean);
		      err |= clSetKernelArg(kernel_reduce, 1, sizeof(cl_mem), &d_data);
		      err |= clSetKernelArg(kernel_reduce, 2, sizeof(int), &m);
		      err |= clSetKernelArg(kernel_reduce, 3, sizeof(int), &n);
		      if (err != CL_SUCCESS) {
		          printf("Error: Failed to set reduce kernel arguments! %d\n", err);
		          printf("Test failed\n");
		          return EXIT_FAILURE;
		      }


		      cl_event readevent_reduce;
		      local_reduce[0]  = n;
		      local_reduce[1]  = m;
		      global_reduce[0] = ceil(n/(float)local_reduce[0])*local_reduce[0];
		      global_reduce[1] = ceil(m/(float)local_reduce[1])*local_reduce[1];
		      err = clEnqueueNDRangeKernel(commands, kernel_reduce, 2, NULL,
		                                   (size_t*)&global_reduce, (size_t*)&local_reduce, 0, NULL, NULL);

		      if (err) {
		               printf("Error: Failed to execute kernel! %d\n", err);
		               printf("Test failed\n");
		               return EXIT_FAILURE;
		           }


		      err = 0;
		      err  = clSetKernelArg(kernel_covar, 0, sizeof(cl_mem), &d_covar);
		      err |= clSetKernelArg(kernel_covar, 1, sizeof(cl_mem), &d_data);
		      err |= clSetKernelArg(kernel_covar, 2, sizeof(int), &m);
		      err |= clSetKernelArg(kernel_covar, 3, sizeof(int), &n);
		      if (err != CL_SUCCESS) {
		          printf("Error: Failed to set reduce kernel arguments! %d\n", err);
		          printf("Test failed\n");
		          return EXIT_FAILURE;
		      }


		      cl_event readevent_covar;
		      local_cover[0]  = m ;
		      global_cover[0] = ceil(m/(float)local_cover[0])*local_cover[0];
		      err = clEnqueueNDRangeKernel(commands, kernel_covar, 1, NULL,
		                                   (size_t*)&global_cover, (size_t*)&local_cover, 0, NULL, NULL);

		      if (err) {
		               printf("Error: Failed to execute kernel! %d\n", err);
		               printf("Test failed\n");
		               return EXIT_FAILURE;
		           }


	      // Read back the results from the device to verify the output
	       //



	       cl_event readevent_read;
	       err = clEnqueueReadBuffer( commands, d_mean, CL_TRUE, 0, sizeof(DATA_TYPE) * m, h_mean, 0, NULL, NULL );
	       if (err != CL_SUCCESS) {
	               printf("Error: Failed to read output array! %d\n", err);
	               printf("Test failed\n");
	               return EXIT_FAILURE;
	           }

	       err = clEnqueueReadBuffer( commands, d_data, CL_TRUE, 0, sizeof(DATA_TYPE) * n*m, h_data, 0, NULL, NULL);
	       if (err != CL_SUCCESS) {
	               printf("Error: Failed to read output array! %d\n", err);
	               printf("Test failed\n");
	               return EXIT_FAILURE;
	       }

	       err = clEnqueueReadBuffer( commands, d_covar, CL_TRUE, 0, sizeof(DATA_TYPE) * m*m, h_covar, 0, NULL, NULL);
	       if (err != CL_SUCCESS) {
	               printf("Error: Failed to read output array! %d\n", err);
	               printf("Test failed\n");
	               return EXIT_FAILURE;
	       }

	       //clWaitForEvents(1, &readevent_read);


	   	clFlush(commands);
	   	clFinish(commands);


	       mean_golden(h_mean_golden, h_data_golden, h_covar_golden, float_n, m, n);

	       for (int i = 0; i < m; i++) {
	    	   DATA_TYPE gold=h_mean_golden[i];
	    	   DATA_TYPE hw=h_mean[i];
	    	   DATA_TYPE diff = fabs(gold-hw);
	    	   if (diff > 0.1) {
	    		   printf("Error at mean %d golden= %f, hw=%f\n", i, gold, hw);
	    		   break;
	    	   }
	       }
	       for (int i = 0; i < m*n; i++) {
	    	   DATA_TYPE gold=h_data_golden[i];
	    	   DATA_TYPE hw=h_data[i];
	    	   DATA_TYPE diff = fabs(gold-hw);
	    	   if (diff > 0.1) {
	    		   printf("Error at data %d golden= %f, hw=%f\n", i, gold, hw);
	    		   break;
	    	   }
	       }

	       for (int i = 0; i < m*m; i++) {
	    	   DATA_TYPE gold=h_covar_golden[i];
	    	   DATA_TYPE hw=h_covar[i];
	    	   DATA_TYPE diff = fabs(gold-hw);
	    	   if (diff > 0.1) {
	    		   printf("Error at covar %d golden= %f, hw=%f\n", i, gold, hw);
	    		   break;
	    	   }
	       }

	       printf("From main: Bye Covariance\n");
	       printf("From main: ====================\n");
	       return 0;
}

void mean_golden(DATA_TYPE *mean, DATA_TYPE *data,DATA_TYPE *covar, DATA_TYPE float_n, int m, int n) {

	for(int j = 0; j < m; j++) {
		for(int i = 0; i < n; i++) {
			mean[j] += data[i * m + j];
		}
		mean[j] /= (DATA_TYPE)float_n;
	}


	for(int j = 0; j < m; j++) {
		for(int i = 0; i < n; i++) {
			data[i * m + j] -= mean[j];
		}
	}


	for(int j1 = 0; j1 < m; j1++) {
		for (int j2 = j1; j2 < m; j2++) {
			covar[j1*m + j2] = 0.0;
			for(int i = 0; i < n; i++) {
				covar[j1 * m + j2] += data[i * m + j1] * data[i * m + j2];
			}
			covar[j2 * m + j1] = covar[j1 * m + j2];
		}
	}
}


