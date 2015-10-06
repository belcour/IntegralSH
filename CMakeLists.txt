cmake_minimum_required (VERSION 3.2)
project (AxialMoments)

set(CMAKE_BUILD_TYPE DEBUG)

include_directories ("include" "externals/Eigen" "externals/glm")

add_executable (TestMnVsMC   tests/MomentsVsMC.cpp   include/AxialMoments.hpp)
add_executable (TestMnVsArvo tests/MomentsVsArvo.cpp include/AxialMoments.hpp)
add_executable (TestMnSign   tests/MomentsSign.cpp   include/AxialMoments.hpp)
add_executable (TestMnEqual  tests/MomentsEquals.cpp include/AxialMoments.hpp)
target_compile_features(TestMnVsArvo PRIVATE cxx_range_for)
target_compile_features(TestMnVsMC   PRIVATE cxx_range_for)
target_compile_features(TestMnSign   PRIVATE cxx_range_for)
target_compile_features(TestMnEqual  PRIVATE cxx_range_for)

enable_testing()
add_test(TestMnVsMC   TestMnVsMC)
add_test(TestMnVsArvo TestMnVsArvo)
add_test(TestMnSign   TestMnSign)
add_test(TestMnEqual  TestMnEqual)
