

__kernel void __attribute__ ((reqd_work_group_size(2048, 1, 1)))
vector_addition(__global float* arrayA, __global float* arrayB, __global float* arrayC) {

	int globalIndex = get_global_id(0);


	arrayC[globalIndex] = arrayA[globalIndex] + arrayB[globalIndex];

}
