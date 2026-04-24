add_executable(hello_mpi_cuda main.cu)
target_link_libraries(hello_mpi_cuda PRIVATE MPI::MPI_CXX CUDA::cudart)
set_target_properties(hello_mpi_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
