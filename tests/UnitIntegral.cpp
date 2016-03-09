// STL includes
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <utility>

// Local includes
//#define USE_SPARSE_EIGEN
#include "Tests.hpp"
#include "SH.hpp"
#include "SphericalHarmonics.hpp"
#include "SphericalIntegration.hpp"
#include "DirectionsSampling.hpp"

// GLM include
#include <glm/glm.hpp>


bool CheckNormalization(const Eigen::VectorXf& clm, const std::vector<Vector>& basis) {

   Triangle tri;

   float shI = 0.0f;

   // Upper hemisphere
   tri = Triangle(Vector(0,0,1), Vector(0,1,0), Vector(1,0,0));
   shI += clm.dot(AxialMoments<Triangle, Vector>(tri, basis));

   tri = Triangle(Vector(0,0,1), Vector(1,0,0), Vector(0,-1,0));
   shI += clm.dot(AxialMoments<Triangle, Vector>(tri, basis));

   tri = Triangle(Vector(0,0,1), Vector(0,-1,0), Vector(-1,0,0));
   shI += clm.dot(AxialMoments<Triangle, Vector>(tri, basis));

   tri = Triangle(Vector(0,0,1), Vector(-1,0,0), Vector(0,1,0));
   shI += clm.dot(AxialMoments<Triangle, Vector>(tri, basis));

   // Lower hemisphere
   tri = Triangle(Vector(0,0,-1), Vector(1,0,0), Vector(0,1,0));
   shI += clm.dot(AxialMoments<Triangle, Vector>(tri, basis));

   tri = Triangle(Vector(0,0,-1), Vector(0,1,0), Vector(-1,0,0));
   shI += clm.dot(AxialMoments<Triangle, Vector>(tri, basis));

   tri = Triangle(Vector(0,0,-1), Vector(-1,0,0), Vector(0,-1,0));
   shI += clm.dot(AxialMoments<Triangle, Vector>(tri, basis));

   tri = Triangle(Vector(0,0,-1), Vector(0,-1,0), Vector(1,0,0));
   shI += clm.dot(AxialMoments<Triangle, Vector>(tri, basis));

   bool check = closeTo(shI, 0.0f);
   if(!check) { std::cout << "Error, lost in precision: I=" << shI; }

   return check;
}

int main(int argc, char** argv) {

   int nb_fails = 0;
   int maxorder = 8;
   int maxsize  = (maxorder+1)*(maxorder+1);
   // Precompute the set of ZH directions
   const auto basis = SamplingFibonacci<Vector>(2*maxorder+1);
   //const auto basis = SamplingBlueNoise<Vector>(2*maxorder+1);
   std::cout << "Done sampling enough directions" << std::endl;

   // Get the Zonal weights matrix and the Zlm -> Ylm conversion matrix
   // and compute the product of the two: `Prod = A x Zw`.
   const auto ZW = ZonalWeights<Vector>(basis);
   const auto Y  = ZonalExpansion<SH, Vector>(basis);
   const auto A  = computeInverse(Y);
   const auto Prod = A*ZW;
   std::cout << "Done with precomputing the matrix" << std::endl;

   for(int order=1; order<=maxorder; ++order) {

      // Loop for each SH coeff on this band
      int size  = (order+1)*(order+1);
      for(int i=order*order; i<size; ++i) {
         // Set the ith coeffcient to one
         Eigen::VectorXf clm = Eigen::VectorXf::Zero(maxsize);
         clm[i] = 1.0f;

         // Perform integral and check the result
         bool check = CheckNormalization(clm.transpose()*Prod, basis);
         if(!check) {
            std::cout << " for i=" << i << " at order=" << order << std::endl;
            ++nb_fails;
         }
      }
   }

   if(nb_fails) {
      return EXIT_FAILURE;
   } else {
      return EXIT_SUCCESS;
   }
}
