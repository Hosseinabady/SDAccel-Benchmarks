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




__kernel __attribute__ ((reqd_work_group_size(1, 1, 1)))
void covar_kernel(__global DATA_TYPE *data, __global DATA_TYPE *covar, int m, int n)
{

	__local DATA_TYPE localData[N*M];//  __attribute__((xcl_array_partition(block, M, 1)));
	__local DATA_TYPE localMean[M];//  __attribute__((xcl_array_partition(complete, 1)));

	async_work_group_copy(localData, data, N*M, 0);

	int j1 = get_global_id(0);
	__local DATA_TYPE localCovar[M*M];// __attribute__((xcl_array_partition(cyclic, M, 1)));


	DATA_TYPE data_buffer[M]; // __attribute__((xcl_array_partition(complete, 1)));



	__attribute__((xcl_pipeline_loop))
	for (int j = 0; j < M; j++) {
		localMean[j] = 0;
	}


	for (int j = 0; j < N; j++) {
	__attribute__((xcl_pipeline_loop))
		for (int k = 0; k < M; k++) {
			localData[j*M+k] = data[j*M+k];
		}
	}





	for (int j = 0; j < N; j++) {
		__attribute__((xcl_pipeline_loop))
		for(int i = 0; i < n; i++) {
			localMean[j] += localData[i*M+j];
		}
		localMean[j] /= (DATA_TYPE)n;
	}



	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
			__attribute__((xcl_pipeline_loop))
			for (int k = 0; k < M; k++) {
				if (i==0)
					localCovar[k*M+j] = 0;
				if (j==0)
					data_buffer[k] = localData[i*M+k]-localMean[k];
				localCovar[k*M+j]  += data_buffer[j]*data_buffer[k];

			}
			if (j>i) {
				localCovar[j*M+i] = localCovar[i*M+j];
			}
		}
	}



	async_work_group_copy(covar, localCovar, M*M, 0);




}
