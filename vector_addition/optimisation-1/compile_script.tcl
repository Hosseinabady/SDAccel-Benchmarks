# SDAccel Application Example - vector_addition
create_solution -name vector_addition_solution -force

# Target a Xilinx FPGA board

add_device -vbnv "xilinx:adm-pcie-7v3:1ddr:3.0"



# Execution arguments for the application
set args "vector_addition_bin.xclbin"


# Host Compiler Flags
set_property -name host_cflags -value "-g -O0 -std=c++0x -I$::env(PWD)" -objects [current_solution]

# Host source files
add_files vector_addition.cpp vector_addition.h common.cpp


# Kernel Definition


create_kernel read_data_kernel -type "clc"
add_files -kernel [ get_kernels read_data_kernel ] vector_addition.cl
set_property max_memory_ports true [ get_kernels read_data_kernel ] 
set_property memory_port_data_width 512 [ get_kernels read_data_kernel ]

create_kernel add_data_kernel -type "clc"
add_files -kernel [ get_kernels add_data_kernel ] vector_addition.cl


create_kernel write_data_kernel -type "clc"
add_files -kernel [ get_kernels write_data_kernel ] vector_addition.cl
set_property max_memory_ports true [ get_kernels write_data_kernel ] 
set_property memory_port_data_width 512 [ get_kernels write_data_kernel ]


create_opencl_binary vector_addition_bin
set_property region "OCL_REGION_0" [ get_opencl_binary vector_addition_bin ]


# Define Binary Containers
create_opencl_binary vector_addition_bin
set_property region "OCL_REGION_0" [get_opencl_binary vector_addition_bin]

create_compute_unit -opencl_binary [get_opencl_binary vector_addition_bin] -kernel [get_kernels read_data_kernel]       -name cu_read
create_compute_unit -opencl_binary [get_opencl_binary vector_addition_bin] -kernel [get_kernels add_data_kernel]        -name cu_addition
create_compute_unit -opencl_binary [get_opencl_binary vector_addition_bin] -kernel [get_kernels write_data_kernel]      -name cu_write


# Compile the design for CPU based emulation
#compile_emulation -flow cpu -opencl_binary [get_opencl_binary vector_addition_bin]
#run_emulation -flow cpu -args $args


# Create estimated resource usage and latency report
#report_estimate

# Compile the design for Hardware Emulation
compile_emulation -flow hardware -opencl_binary [get_opencl_binary vector_addition_bin]
run_emulation -flow hardware -args $args


report_estimate

# Compile the design for execution on the FPGA board
#build_system

# Create the board deployment package for the application
#package_system