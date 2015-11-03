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
#include "SphericalIntegration.hpp"

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

struct CosPowFunctor {
   // Constructor
   int _exp;
   inline CosPowFunctor(int exp=1) : _exp(exp) {}

   // Operator function
   inline float operator()(const Vector& w) const {
      return pow(glm::clamp(w.z, 0.0f, 1.0f), _exp);
   }
};

/* Test the SH projection with respect to a diffuse functor: f(w) = (w·n)+
 */
int TestPhongProjection(int order = 5, int exp = 1, float Epsilon = 1.0E-3f) {

   std::cout << "Test of the project of a cosine power to SH" << std::endl;
   std::cout << "  + SH max order = " << order << std::endl;
   std::cout << "  + Cosine power = " << exp << std::endl;

   const int msize = (order+1)*(order+1);
   int nb_fails    = 0;

   const CosPowFunctor f(exp);
   const std::vector<Vector>  basis = SamplingBlueNoise<Vector>(msize);
   const Eigen::VectorXf      clm   = ProjectToSH<CosPowFunctor, Vector, SH>(f, basis);

   for(auto& w : basis) {

      const Eigen::VectorXf ylm = SH::FastBasis(w, order);

      const float pi = ylm.dot(clm);
      const float fi = f(w);

      if(!closeTo(fi, pi)) {
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

      if(!closeTo(fi, pi)) {
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

/* Compute the integral of the spherical function defined by the SH
 * coefficients `clm` over the spherical triangle using MC.
 */
std::pair<float,float> MonteCarloSH(const Eigen::VectorXf& clm,
                                    const Triangle& triangle) {

   static std::mt19937 gen(0);
   static std::uniform_real_distribution<float> dist(0.0,1.0);

   const int order = sqrt(clm.size());

   // Number of MC samples
   const int M = 10000000;
   float mean = 0.0f;
   float var  = 0.0f;
   for(int k=0; k<M; ++k) {

#ifdef USE_TRIANGLE_SAMPLING
      float pdf;
      const Vector d = SampleSphericalTriangle(triangle, pdf);
#else // UNIFORM SAMPLING
      const float pdf = 1.0f / (4.0f*M_PI);
      const Vector d  = Sample();
#endif

      const auto ylm = SHEvalFast<Vector>(d, order-1);

      if(HitTriangle(triangle, d)) {
         const auto val = ylm.dot(clm) / pdf;
         mean += val;
         var  += val*val;
      }

   }

   mean /= M;
   var   = var / (M-1) - mean*mean;
   return std::pair<float,float>(mean, 5.0f*sqrt(var/M));
}

int main(int argc, char** argv) {
   int nb_fails = 0;

   int order = 5;
   int exp   = 1;

   // Test the diffuse project (exponent = 1)
   nb_fails += TestPhongProjection(order, exp);

   // Test for low exponnent phong
   exp = 3;
   nb_fails += TestPhongProjection(order, exp);

   // Test for mid exponnent phong
   order = 15;
   exp   = 10;
   nb_fails += TestPhongProjection(order, exp);

   if(nb_fails > 0) {
      return EXIT_FAILURE;
   } else {
      return EXIT_SUCCESS;
   }
}
