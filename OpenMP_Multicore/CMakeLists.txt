cmake_minimum_required(VERSION 3.16)
project(cylinder_solver_openmp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenMP REQUIRED)

file(GLOB SOLVER_CPP CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB SOLVER_C CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.c")
set_source_files_properties(${SOLVER_C} PROPERTIES LANGUAGE CXX)
add_executable(cylinder_solver ${SOLVER_CPP} ${SOLVER_C})
target_link_libraries(cylinder_solver PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(cylinder_solver PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang>:-O2>)

file(COPY input DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/data/restart")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/transient/data")

enable_testing()
add_test(NAME smoke COMMAND cylinder_solver 12 2)
set_tests_properties(smoke PROPERTIES
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/tests/smoke"
    ENVIRONMENT "OMP_NUM_THREADS=2")
