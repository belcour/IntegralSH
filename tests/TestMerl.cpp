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

int TestMerlProjectionSlice(const std::string& filename) {

   int order = 18;
   int elev  = 90;


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

   // Get the SH expansion of rho as a list of SH vectors each of a particular
   // incidence. This is the method of Sillion et al.
   std::vector<Eigen::MatrixXf> matrices;
   matrices.reserve(elev);
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

int TestMerlProjectionMatrix(const std::string& filename,
                             int order = 10) {

   // Constants
   const int size = SH::Terms(order);

   // Load the BRDF
   MerlBRDF brdf;
   brdf.read_brdf(filename);

   // Values
   std::mt19937 gen(0);
   std::uniform_real_distribution<float> dist(0.0,1.0);
   std::vector<Eigen::MatrixXf> cijs(3, Eigen::MatrixXf::Zero(size, size));

   for(int i=0; i<M; ++i) {
      // Sample the output direction
      Vector wo;
      wo.z = 2.0*dist(gen) - 1.0;
      const float z2 = wo.z*wo.z;
      if(wo.z < 0.0) { continue; }
      const float phi = 2.0*M_PI*dist(gen);
      wo.x = sqrt(1.0f-z2) * cos(phi);
      wo.y = sqrt(1.0f-z2) * sin(phi);

      // Sample the input direction
      Vector wi;
      wi.z = 2.0*dist(gen) - 1.0;
      const float z2 = wi.z*wi.z;
      if(wi.z < 0.0) { continue; }
      const float phi = 2.0*M_PI*dist(gen);
      wi.x = sqrt(1.0f-z2) * cos(phi);
      wi.y = sqrt(1.0f-z2) * sin(phi);

      // Evaluate the BRDF value
      const auto rgb = brdf.value<RGB, Vector>(wi, wo);
      const auto ylmo = SH::FastBasis(wo, order);
      const auto ylmi = SH::FastBasis(wi, order);

      // Enforce reciprocity
      const MatrixXf mat1 = ylmo * ylmi.transpose();
      const MatrixXf mat2 = ylmi * ylmo.transpose();
      cijs += 0.5 * rgb[0]*(mat1 + mat2);
      cijs += 0.5 * rgb[1]*(mat1 + mat2);
      cijs += 0.5 * rgb[2]*(mat1 + mat2);
   }
   cij *= 16.0*M_PI*M_PI / float(M);


   // Print values
   const float theta_i = 0.0;
   const Vector wi(sin(thetai), 0, cos(thetai));
   const auto ylmi = SH::FastBasis(wi, order);
   Vector wo;
   for(int i=0; i<elev; ++i) {
      const float theta = 0.5*M_PI * i / float(elev);
      wo.x = sin(theta);
      wo.y = 0;
      wo.z = cos(theta);

      // Ref
      const auto RGB = brdf.value<RGB, Vector>(wi, wo);
      const float R = RGB[0];
      const float G = RGB[1];
      const float B = RGB[2];

      // SH expansion
      const auto ylmo = SH::FastBasis(wo, order);
      const float r = ylmi.dot(ylmo * cijs[0]);
      const float g = ylmi.dot(ylmo * cijs[1]);
      const float b = ylmi.dot(ylmo * cijs[2]);

      std::cout << theta << "\t" << r << "\t" << g << "\t" << b
                         << "\t" << R << "\t" << G << "\t" << B
                         << std::endl;
   }

   int nb_fails = 0;
   return nb_fails;
}

int main(int argc, char** argv) {

   int nb_fails = 0;

   // Load an example
   nb_fails += TestMerlProjectionSlice("gold-paint.binary");
   nb_fails += TestMerlProjectionMatrix("gold-paint.binary");

   if(nb_fails > 0) {
      return EXIT_FAILURE;
   } else {
      return EXIT_SUCCESS;
   }
}
