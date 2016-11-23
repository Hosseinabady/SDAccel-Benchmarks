create_solution -name mx_ad_solution

add_device -vbnv "xilinx:adm-pcie-7v3:1ddr:2.1"
set_property -name host_cflags -value {-Wall -D FPGA_DEVICE} -objects [ current_solution ]

add_files vector_addition.cpp vector_addition.h xcl.c xcl.h
create_opencl_binary vector_add
set_property region "OCL_REGION_0" [ get_opencl_binary vector_add ]

create_kernel "vector_addition" -type "clc"
add_files -kernel [ get_kernels vector_addition ] vector_addition.cl

create_compute_unit -opencl_binary [ get_opencl_binary vector_add ] -kernel [ get_kernels "vector_addition" ] -name "k1_1"


build_system
package_system
