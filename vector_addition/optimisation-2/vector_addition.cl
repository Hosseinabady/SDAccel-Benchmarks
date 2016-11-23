
#include "vector_addition.h"




pipe  DATA_TYPE16 pa __attribute__((xcl_reqd_pipe_depth(PIPE_DEPTH)));
pipe  DATA_TYPE16 pb __attribute__((xcl_reqd_pipe_depth(PIPE_DEPTH)));
pipe  DATA_TYPE16 pc __attribute__((xcl_reqd_pipe_depth(PIPE_DEPTH)));

__kernel void __attribute__ ((reqd_work_group_size(WORK_GROUP_SIZE, 1, 1)))
read_data_kernel(__global DATA_TYPE16* vectorA, __global DATA_TYPE16* vectorB) {

	int localIndex  = get_local_id(0);
	int globalIndex = get_global_id(0);
	DATA_TYPE16 a = vectorA[globalIndex];
	DATA_TYPE16 b = vectorB[globalIndex];

	__attribute__((xcl_pipeline_loop))
	for (int i = 0; i < THREAD_IN_WORK_ITEM_SIZE; i++) {
		globalIndex = localIndex*THREAD_IN_WORK_ITEM_SIZE + i;
//		printf("from read_data_kernel iteration %d \n", globalIndex);

		a = vectorA[globalIndex];
		b = vectorB[globalIndex];

		write_pipe_block(pa, &a);
		write_pipe_block(pb, &b);
	}

}

__kernel void __attribute__ ((reqd_work_group_size(WORK_GROUP_SIZE, 1, 1)))
add_data_kernel(int inc) {

	int localIndex  = get_local_id(0);
	int globalIndex = get_global_id(0);



	__attribute__((xcl_pipeline_loop))
	for (int i = 0; i < THREAD_IN_WORK_ITEM_SIZE; i++) {
		globalIndex = localIndex*THREAD_IN_WORK_ITEM_SIZE + i;
		DATA_TYPE16 a;
		DATA_TYPE16 b;
		DATA_TYPE16 c;
		globalIndex = i;
//		printf("from addition_data_kernel iteration %d \n", globalIndex);

		read_pipe_block(pa, &a);
		read_pipe_block(pb, &b);

		c = a + b;

		write_pipe_block(pc, &c);
	}
}

__kernel void __attribute__ ((reqd_work_group_size(WORK_GROUP_SIZE, 1, 1)))
write_data_kernel(__global DATA_TYPE16* vectorC) {
	int localIndex  = get_local_id(0);
	int globalIndex = get_global_id(0);


	DATA_TYPE16 c;
	__attribute__((xcl_pipeline_loop))
	for (int i = 0; i < THREAD_IN_WORK_ITEM_SIZE; i++) {
		globalIndex = localIndex*THREAD_IN_WORK_ITEM_SIZE + i;
//		printf("from write_data_kernel iteration %d \n", globalIndex);
		read_pipe_block(pc, &c);

		vectorC[globalIndex] = c;
	}
}
