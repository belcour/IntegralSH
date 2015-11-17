// STL includes
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <utility>

// Local includes
#include "Tests.hpp"
#include "Merl.hpp"
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

struct MerlSlice {

   const MerlBRDF& brdf;
   const Vector&   wi;

   // Constructor
   MerlSlice(const MerlBRDF& brdf, const Vector& wi) : brdf(brdf), wi(wi) {}

   // Functor capabilities
   Eigen::Vector3f operator()(const Vector& wo) const {
      return brdf.value<Eigen::Vector3f, Vector>(wi, wo);
   }
};

int TestMerlProjection(const std::string& filename) {

   int order = 18;
   int elev  = 90;

   std::vector<Eigen::MatrixXf> matrices;
   matrices.reserve(elev);

   // Load the BRDF
   MerlBRDF brdf;
   brdf.read_brdf(filename);

   // Print the values
   std::vector<Vector> idirs;
   for(int i=0; i<elev; ++i) {
      const float theta = 0.5*M_PI * i / float(elev);
      Vector wo(sin(theta), 0, cos(theta));
      idirs.push_back(wo);
   }

   /*
   std::vector<Vector> odirs = SamplingFibonacci<Vector>(10000);
   for(auto& wi : idirs) {

      MerlSlice slice(brdf, wi);
      Eigen::MatrixXf matrix(SH::Terms(order), 3);

      for(auto& wo : odirs) {
         // Do not print value for below horizon vectors
         if(wo.z > 0.0) {
            const auto rgb = slice(wo);
            const auto ylm = SH::FastBasis(wo, order);

            matrix.col(0) += rgb[0]*ylm;
            matrix.col(1) += rgb[1]*ylm;
            matrix.col(2) += rgb[2]*ylm;
         }
      }

      matrix *= 4.0*M_PI / float(odirs.size());
      matrices.push_back(matrix);
   }*/
   matrices = brdf.projectToSH<Eigen::Vector3f, Vector, SH>(elev, order);


   // Print values
   int col = 0;
   const float thetai = 0.5*M_PI * col/float(elev);
   Vector wi(sin(thetai), 0, cos(thetai));
   Vector wo;
   for(int i=0; i<elev; ++i) {
      const float theta = 0.5*M_PI * i / float(elev);
      wo.x = sin(theta);
      wo.y = 0;
      wo.z = cos(theta);

      const auto ylm = SH::FastBasis(wo, order);
      std::cout << theta << " " << matrices[col].col(0).dot(ylm)
                         << " " << brdf.value<Eigen::Vector3f, Vector>(wi, wo).transpose() << std::endl;
   }

   return 0;
}


int main(int argc, char** argv) {

   int nb_fails = 0;

   // Load an example
   nb_fails += TestMerlProjection("gold-paint.binary");

   if(nb_fails > 0) {
      return EXIT_FAILURE;
   } else {
      return EXIT_SUCCESS;
   }
}
