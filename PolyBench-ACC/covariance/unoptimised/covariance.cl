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

__kernel __attribute__ ((reqd_work_group_size(M, 1, 1)))
void mean_kernel(__global DATA_TYPE *mean, __global DATA_TYPE *data, DATA_TYPE float_n, int m, int n)
{
	int j = get_global_id(0);

	mean[j] = 0.0;


	int i;
	for(i = 0; i < n; i++) {
		mean[j] += data[i * m + j];
	}
	mean[j] /= (DATA_TYPE)float_n;


}

__kernel __attribute__ ((reqd_work_group_size(N, M, 1)))
void reduce_kernel(__global DATA_TYPE *mean, __global DATA_TYPE *data, int m, int n)
{
	int i = get_global_id(0);
	int j = get_global_id(1);


	data[i * m + j] -= mean[j];
}

__kernel __attribute__ ((reqd_work_group_size(M, 1, 1)))
void covar_kernel(__global DATA_TYPE *covar, __global DATA_TYPE *data, int m, int n)
{
	int j1 = get_global_id(0);
	int i, j2;


	if (j1 < m)
	{
		for (j2 = j1; j2 < m; j2++)
		{
			covar[j1*m + j2] = 0.0;
			for(i = 0; i < n; i++)
			{
				covar[j1 * m + j2] += data[i * m + j1] * data[i * m + j2];
			}
			covar[j2 * m + j1] = covar[j1 * m + j2];
		}
	}
}
