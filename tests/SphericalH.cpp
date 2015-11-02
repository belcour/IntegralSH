// STL includes
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <utility>

// Local includes
#include "Tests.hpp"
#include "SH.hpp"
#include "SphericalHarmonics.hpp"
#include "DirectionsSampling.hpp"

// GLM include
#include <glm/glm.hpp>

struct SH {

   // Inline for FastBasis
   static inline Eigen::VectorXf FastBasis(const Vector& w, int lmax) {
      return SHEvalFast<Vector>(w, lmax);
   }

   // Inline for Terms
   static inline int Terms(int band) {
      return SHTerms(band);
   }

   // Inline for Index
   static inline int Index(int l, int m) {
      return SHIndex(l, m);
   }
};

/* Test the SH projection with respect to a diffuse functor: f(w) = (w·n)+
 */
int TestDiffuseProjection(int order = 5, float Epsilon = 1.0E-3f) {

   struct DiffuseFunctor {
      // Constructor
      inline DiffuseFunctor() {}

      // Operator function
      inline float operator()(const Vector& w) const {
         return glm::clamp(w.z, 0.0f, 1.0f);
      }
   };

   const int msize = (order+1)*(order+1);
   int nb_fails    = 0;

   const DiffuseFunctor f;
   const std::vector<Vector>  basis = SamplingBlueNoise<Vector>(msize);
   const Eigen::VectorXf      clm   = ProjectToSH<DiffuseFunctor, Vector, SH>(f, basis);

   for(auto& w : basis) {

      const Eigen::VectorXf ylm = SH::FastBasis(w, order);

      const float pi = ylm.dot(clm);
      const float fi = f(w);

      if(std::abs(pi - fi) > Epsilon) {
         std::cout << "SH(w) = " << pi << " ≠ " << fi << " f(w), "
                   << "for w = "<< w << std::endl;
         ++nb_fails;
      }
   }

   // Regenerate another set of directions
   const std::vector<Vector> query = SamplingBlueNoise<Vector>(100);
   for(auto& w : basis) {

      const Eigen::VectorXf ylm = SH::FastBasis(w, order);

      const float pi = ylm.dot(clm);
      const float fi = f(w);

      if(std::abs(pi - fi) > Epsilon) {
         std::cout << "SH(w) = " << pi << " ≠ " << fi << " f(w), "
                   << "for w = "<< w << std::endl;
         ++nb_fails;
      }
   }

   if(nb_fails > 0) {
      std::cerr << "Test failed!" << std::endl;
   } else {
      std::cout << "Test success!" << std::endl;
   }
   std::cout << std::endl;

   return nb_fails;
}

int main(int argc, char** argv) {
   int nb_fails = 0;

   nb_fails += TestDiffuseProjection();

   if(nb_fails > 0) {
      return EXIT_FAILURE;
   } else {
      return EXIT_SUCCESS;
   }
}
