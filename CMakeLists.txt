cmake_minimum_required (VERSION 3.2)
project (AxialMoments)

include_directories ("include" "externals/Eigen" "externals/glm")

add_executable (Test tests/TestMoments.cpp)
target_compile_features(Test PRIVATE cxx_range_for)

enable_testing()
add_test(MomentsTest Test)
