# SDAccel Application Example - Covariance
create_solution -name covariance_solution -force

# Target a Xilinx FPGA board

add_device -vbnv "xilinx:adm-pcie-7v3:1ddr:3.0"



# Execution arguments for the application
set args "covariance_bin.xclbin"


# Host Compiler Flags
set_property -name host_cflags -value "-g -O0 -std=c++0x -I$::env(PWD)" -objects [current_solution]

# Host source files
add_files covariance.cpp covariance.h common.cpp


# Kernel Definition


create_kernel covar_kernel -type "clc"
add_files -kernel [ get_kernels covar_kernel ] covariance.cl


# Define Binary Containers
create_opencl_binary covariance_bin
set_property region "OCL_REGION_0" [get_opencl_binary covariance_bin]


create_compute_unit -opencl_binary [get_opencl_binary covariance_bin] -kernel [get_kernels covar_kernel]    -name cu_covar



# Compile the design for CPU based emulation
#compile_emulation -flow cpu -opencl_binary [get_opencl_binary covariance_bin]
#run_emulation -flow cpu -args $args


# Create estimated resource usage and latency report
#report_estimate

# Compile the design for Hardware Emulation
#compile_emulation -flow hardware -opencl_binary [get_opencl_binary covariance_bin]
#run_emulation -flow hardware -args $args


#report_estimate

# Compile the design for execution on the FPGA board
build_system

# Create the board deployment package for the application
package_system
