#ifndef __VECTOR_ADDITION_h__
#define __VECTOR_ADDITION_h__

#define DATA_LENGTH  (2048*2084*8)
#define DATA_TYPE    float
#define DATA_TYPE16  float16

#define PIPE_DEPTH 16

#define WORK_GROUP_SIZE 1

#define THREAD_IN_WORK_ITEM_SIZE (DATA_LENGTH/16)


#endif // __VECTOR_ADDITION_h__
