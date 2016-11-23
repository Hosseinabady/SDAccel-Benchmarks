/* File: covariance.cl
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
* File name : covariance.cl.cl
* author    : Mohammad hosseinabady mohammad@hosseinabady.com
* date      : 12 November 2016
* blog: https://highlevel-synthesis.com/
*/

#include "covariance.h"

typedef unsigned int u32;

union data_element{
	DATA_TYPE    d[16];
	DATA_TYPE16  d16;
};

typedef union data_element data_element_type;


//global DATA_TYPE16 onChipGlobalData[N*M];
//global DATA_TYPE16 onChipGlobalMean[1024];


__kernel __attribute__ ((reqd_work_group_size(1, 1, 1)))
void mean_kernel(__global DATA_TYPE16* data, __global DATA_TYPE16* mean, __global DATA_TYPE16* covar, int m, int n) {

	data_element_type d_e;
	DATA_TYPE16 d;

	DATA_TYPE16 onChipGlobalData[N*M/16];
	DATA_TYPE16 onChipGlobalMean[M/16];


	__attribute__((xcl_pipeline_loop))
	for (u32 j = 0; j < M/16; j++) {
		onChipGlobalMean[j] = 0;
	}


	for (u32 i = 0; i < N; i++) {
		__attribute__((xcl_pipeline_loop))
		for(u32 j = 0; j < M/16; j++) {
			d = data[i*M/16+j];

			onChipGlobalMean[j]       += d;
			onChipGlobalData[i*M/16+j] = d;
		}
	}


	__attribute__((xcl_pipeline_loop))
	for (u32 j = 0; j < M/16; j++) {
		onChipGlobalMean[j] = onChipGlobalMean[j]/n;
		mean[j] = onChipGlobalMean[j];
	}


	__local DATA_TYPE16 localCovar[M*M/16];// __attribute__((xcl_array_partition(cyclic, 16, 1)));


	DATA_TYPE data_buffer[M] __attribute__((xcl_array_partition(cyclic, 16, 1)));


	for (u32 i = 0; i < N; i++) {
		for (u32 j = 0; j < M; j++) {
			__attribute__((xcl_pipeline_loop))
			for (u32 k = 0; k < M/16; k++) {
				if (i==0)
					localCovar[j*M/16+k] = 0;

				if (j==0) {

					d_e.d16 = onChipGlobalData[i*M/16+k]-onChipGlobalMean[k];
					__attribute__((opencl_unroll_hint))
					for (u32 index = 0; index < 16; index++) {
						data_buffer[k*16+index] = d_e.d[index];
					}
				}
				data_element_type d_e;
				__attribute__((opencl_unroll_hint))
				for (u32 index = 0; index < 16; index++) {
					d_e.d[index] = data_buffer[j]*data_buffer[k*16+index];
				}
				localCovar[j*M/16+k] += d_e.d16;
			}
		}
	}


	for (u32 i = 0; i < M; i++) {
		__attribute__((xcl_pipeline_loop))
		for(u32 j = 0; j < M/16; j++) {
			covar[i*M/16+j] = localCovar[i*M/16+j];
		}
	}

}


