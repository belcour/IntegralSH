cmake_minimum_required (VERSION 3.2)
project (AxialMoments CXX)

# Add platform dependent flags
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
#include (OptimizeForArchitecture)

# For debug build
set(CMAKE_BUILD_TYPE RELEASE)

if ( CMAKE_COMPILER_IS_GNUCC )
   add_definitions ("-Wall -pedantic")
endif ( CMAKE_COMPILER_IS_GNUCC )

include_directories ("include" "externals/eigen" "externals/glm" ".")

# Add dependencies
find_package(Threads REQUIRED)

# Add main test suite
add_executable (TestUtils    tests/TestUtils.cpp     include/Utils.hpp)
add_executable (TestMnVsMC   tests/MomentsVsMC.cpp   include/AxialMoments.hpp)
add_executable (TestMnVsArvo tests/MomentsVsArvo.cpp include/AxialMoments.hpp)
add_executable (TestMnSign   tests/MomentsSign.cpp   include/AxialMoments.hpp)
add_executable (TestMnEqual  tests/MomentsEquals.cpp include/AxialMoments.hpp)
add_executable (TestSpheInt  tests/SphericalInt.cpp  include/AxialMoments.hpp include/SphericalIntegration.hpp tests/SH.hpp)
add_executable (TestSpheHar  tests/SphericalH.cpp    include/AxialMoments.hpp include/SphericalIntegration.hpp tests/SH.hpp)
add_executable (TestRotation tests/TestRotations.cpp)
add_executable (TestProduct  tests/TestProduct.cpp)
add_executable (UnitIntegral tests/UnitIntegral.cpp)
target_link_libraries(TestRotation ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(TestUtils    PRIVATE cxx_range_for)
target_compile_features(TestMnVsArvo PRIVATE cxx_range_for)
target_compile_features(TestMnVsMC   PRIVATE cxx_range_for)
target_compile_features(TestMnSign   PRIVATE cxx_range_for)
target_compile_features(TestMnEqual  PRIVATE cxx_range_for)
target_compile_features(TestSpheInt  PRIVATE cxx_range_for)
target_compile_features(TestSpheHar  PRIVATE cxx_range_for)
target_compile_features(TestRotation PRIVATE cxx_range_for)
target_compile_features(TestProduct  PRIVATE cxx_range_for)
target_compile_features(UnitIntegral PRIVATE cxx_range_for)

enable_testing()
add_test(TestUtils    TestUtils)
add_test(TestSpheHar  TestSpheHar)
add_test(TestMnVsMC   TestMnVsMC)
add_test(TestMnVsArvo TestMnVsArvo)
add_test(TestMnSign   TestMnSign)
add_test(TestMnEqual  TestMnEqual)
add_test(TestSpheInt  TestSpheInt)
add_test(TestRotation TestRotation)
add_test(TestProduct  TestProduct)
add_test(UnitIntegral UnitIntegral)

# Add SH examples
add_executable (Merl2Sh utils/Merl2Sh.cpp)
target_compile_features(Merl2Sh PRIVATE cxx_range_for)
target_compile_options(Merl2Sh INTERFACE -O3 -msse3 -mfpmath=sse)
target_link_libraries(Merl2Sh ${CMAKE_THREAD_LIBS_INIT})
if($ENV{ALTA_DIR})
   include_directories ("$ENV{ALTA_DIR}")
   add_executable (Alta2Sh utils/Alta2Sh.cpp)
   target_compile_features(Alta2Sh PRIVATE cxx_range_for)
   target_compile_options(Alta2Sh INTERFACE -O3 -msse3 -mfpmath=sse -m64 -march=nocona)
   set(LDDIR "$ENV{ALTA_LIB}")
   target_link_libraries(Alta2Sh ${CMAKE_THREAD_LIBS_INIT} -L${LDDIR} 'core' -m64)
endif($ENV{ALTA_DIR})

add_executable (IntegralConvergence utils/IntegralConvergence.cpp)
target_compile_features(IntegralConvergence PRIVATE cxx_range_for)
