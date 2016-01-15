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

/* Compute the projection of the HG phase function to Zonal Harmonics.
 */
Eigen::VectorXf HeyneyeProjection(float g, int order) {

   // Fill the Zonal Coefficients
   Eigen::VectorXf zhG = Eigen::VectorXf::Zero(SH::Terms(order));
   zhG[0] = 1.0f/sqrt(4*M_PI);
   float powg = g;
   for(int k=1; k<order; ++k) {
      const int   index = k + k*k;
      const float factor = sqrt((2*k+1) / (4*M_PI));
      zhG[index] = factor * powg;
      powg *= g;
   }

   return zhG;
}

/* Return the values of the HG function aligned with the Z-axis
 */
float HeyneyeGreenstein(const Vector& w, float g) {
   return ((1 - g*g) / pow(1 + g*g - 2*g*w.z, 3.0/2.0)) / (4.0f*M_PI);
}

/* Compare the projection of the HG function in ZH with its real values.
 */
int TestHeyneyeProjection(float g, int order) {

   int nb_fails = 0;

   // Get the HG approx
   const Eigen::VectorXf zhG = HeyneyeProjection(g, order);

   // Compare to different values of directions
   const auto directions = SamplingFibonacci<Vector>(100);
   for(const auto& w : directions) {
   //for(int nt=0; nt<360; ++nt) {

     // float theta = 2*M_PI * float(nt) / float(360);
     // const Vector w(sin(theta), 0.0, cos(theta));

      const Eigen::VectorXf ylm = SH::FastBasis(w, order);

      const float vSH = ylm.dot(zhG);
      const float vHG = HeyneyeGreenstein(w, g);

      //std::cout << theta << "\t" << vHG << "\t" << vSH << std::endl;
      if(!closeTo(vSH, vHG)) {
         std::cout << "Error: with g=" << g << " and w=";
         std::cout << w << " => " << vSH << " ≠ " << vHG << std::endl;
         nb_fails++;
      }
   }

   return nb_fails;
}

int main(int argc, char** argv) {
   int nb_fails = 0;

   int order = 5;
   int exp   = 1;


   // Test the HG decomposition for trivial case:
   nb_fails += TestHeyneyeProjection(0.0f, 5);
   nb_fails += TestHeyneyeProjection(0.1f, 10);
   nb_fails += TestHeyneyeProjection(0.5f, 18);

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
