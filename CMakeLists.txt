cmake_minimum_required (VERSION 3.2)
project (AxialMoments)

# Add platform dependent flags
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
#include (OptimizeForArchitecture)

# For debug build
set(CMAKE_BUILD_TYPE DEBUG)

include_directories ("include" "externals/eigen" "externals/glm")

# Add main test suite
add_executable (TestUtils    tests/TestUtils.cpp     include/Utils.hpp)
add_executable (TestMnVsMC   tests/MomentsVsMC.cpp   include/AxialMoments.hpp)
add_executable (TestMnVsArvo tests/MomentsVsArvo.cpp include/AxialMoments.hpp)
add_executable (TestMnSign   tests/MomentsSign.cpp   include/AxialMoments.hpp)
add_executable (TestMnEqual  tests/MomentsEquals.cpp include/AxialMoments.hpp)
add_executable (TestSpheInt  tests/SphericalInt.cpp  include/AxialMoments.hpp include/SphericalIntegration.hpp tests/SH.hpp)
add_executable (TestSpheHar  tests/SphericalH.cpp    include/AxialMoments.hpp include/SphericalIntegration.hpp tests/SH.hpp)
add_executable (TestRotation tests/TestRotations.cpp)
target_compile_features(TestUtils    PRIVATE cxx_range_for)
target_compile_features(TestMnVsArvo PRIVATE cxx_range_for)
target_compile_features(TestMnVsMC   PRIVATE cxx_range_for)
target_compile_features(TestMnSign   PRIVATE cxx_range_for)
target_compile_features(TestMnEqual  PRIVATE cxx_range_for)
target_compile_features(TestSpheInt  PRIVATE cxx_range_for)
target_compile_features(TestSpheHar  PRIVATE cxx_range_for)
target_compile_features(TestRotation PRIVATE cxx_range_for)

enable_testing()
add_test(TestUtils    TestUtils)
add_test(TestSpheHar  TestSpheHar)
add_test(TestMnVsMC   TestMnVsMC)
add_test(TestMnVsArvo TestMnVsArvo)
add_test(TestMnSign   TestMnSign)
add_test(TestMnEqual  TestMnEqual)
add_test(TestSpheInt  TestSpheInt)
add_test(TestRotation TestRotation)

# Add SH examples
add_executable (TestMerl  tests/TestMerl.cpp)
target_compile_features(TestMerl PRIVATE cxx_range_for)
add_executable (TestProd  tests/TestProduct.cpp)
target_compile_features(TestProd PRIVATE cxx_range_for)
