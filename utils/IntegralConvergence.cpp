// STL includes
#include <iostream>
#include <vector>

// System includes
#include <sys/time.h>

// Local includes
#include <AxialMoments.hpp>
#include <SphericalIntegration.hpp>
#include <DirectionsSampling.hpp>
#include <Utils.hpp>

// Wrappers and utils from tests
#include <tests/Tests.hpp>

#define NB_TRIALS 1000

void ConvergenceSH(const Eigen::VectorXf& clm,
                   const Triangle& triangle,
                   unsigned int oMin,
                   unsigned int oMax) {

   // Get the order of the provided SH vector and compute the directional
   // sampling of the sphere for rotated ZH/cosines.
   const auto basis = SamplingFibonacci<Vector>(2*oMax+1);

   std::vector<Eigen::MatrixXf> Prods;
   for(unsigned int o=oMin; o<oMax; ++o) {

      const auto cbasis = std::vector<Vector>(&basis[0], &basis[2*o+1]);

      // Get the Zonal weights matrix and the Zlm -> Ylm conversion matrix
      // and compute the product of the two: `Prod = A x Zw`.
      const auto ZW = ZonalWeights<Vector>(cbasis);
      const auto Y  = ZonalExpansion<SH, Vector>(cbasis);
      const auto A  = computeInverse(Y);
      const auto Prod = A*ZW;
      Prods.push_back(Prod);
   }


   for(unsigned int o=oMin; o<oMax; ++o) {

      timeval time;
      gettimeofday(&time, NULL);
      const auto start = (time.tv_sec * 1000.0) + (time.tv_usec / 1000.0);

      float I = 0.0;
      for(int k=0; k<NB_TRIALS; ++k) {

      // Restrict the basis of vector and the SH coefficients of the integrand
      // to only the necessary part
      const auto cbasis = std::vector<Vector>(&basis[0], &basis[2*o+1]);
      const auto ylm = clm.segment(0, SH::Terms(o));

      // Analytical evaluation of the integral of power of cosines for
      // the different basis elements up to the order defined by the
      // number of elements in the basis
      const auto moments = AxialMoments<Triangle, Vector>(triangle, cbasis);

      // Compute the integral using a restricted matrix
      I = ylm.dot(Prods[o-oMin] * moments);
      }

      gettimeofday(&time, NULL);
      const auto end = (time.tv_sec * 1000.0) + (time.tv_usec / 1000.0);

      auto avg_time = (end-start)/NB_TRIALS;

      std::cout << o << "\t" << avg_time << "\t" << I << std::endl;
   }
}


int main(int argc, char** argv) {

   const int oMin = 1;
   const int oMax = 18;
   Eigen::VectorXf clm = Eigen::VectorXf(SH::Terms(oMax));
   clm[0] = 1.0;
   clm[9] = 1.0;

   glm::vec3 A, B, C;
   Triangle tri;
   A = glm::vec3(0.0, 0.0, 1.0);
   B = glm::vec3(0.0, 0.5, 1.0);
   C = glm::vec3(0.5, 0.0, 1.0);
   tri = Triangle(glm::normalize(A), glm::normalize(B), glm::normalize(C));

   ConvergenceSH(clm, tri, oMin, oMax);

   return EXIT_SUCCESS;
}
