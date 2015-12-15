// STL includes
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <random>
#include <utility>

// Local includes
#include "Tests.hpp"
#include "Merl.hpp"
#include "SH.hpp"
#include "Utils.hpp"
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

#include <thread>

struct MerlProjectionThread : public std::thread {

   int order;
   std::vector<Eigen::MatrixXf> cijs;

   MerlProjectionThread(const MerlBRDF* brdf,
                        const std::vector<Vector>* ws,
                        int order, int skip, int nthread) :
      std::thread(&MerlProjectionThread::run, this, brdf, ws, skip, nthread),
      order(order),
      cijs(3, Eigen::MatrixXf::Zero(SH::Terms(order), SH::Terms(order)))
   {}

   void run(const MerlBRDF* brdf,
            const std::vector<Vector>* dirs,
            int skip, int nthread) {

      for(unsigned int i=skip; i<dirs->size(); i+=nthread) {
         const Vector& wo = (*dirs)[i];

         if(skip == 0) {
            std::cout << "Progress: " << i << " / " << dirs->size() << "     \r";
            std::cout.flush();
         }

         // Skip below the horizon configuration
         if(wo.z < 0.0) continue;
         const auto ylmo = SH::FastBasis(wo, order);

         for(unsigned int j=0; j<dirs->size(); ++j) {
            const Vector& wi = (*dirs)[j];
            // Skip below the horizon configuration
            if(wi.z < 0.0) continue;

            // Evaluate the BRDF value
            const auto rgb = brdf->value<Vector, Vector>(wi, wo);
            const auto ylmi = SH::FastBasis(wi, order);

            const Eigen::MatrixXf mat = ylmo * ylmi.transpose();
            cijs[0] += rgb[0]*mat;
            cijs[1] += rgb[1]*mat;
            cijs[2] += rgb[2]*mat;
         }
      }
   }
};

int TestMerlProjectionMatrix(const std::string& filename,
                             int order = 15, int N = 1000) {

   // Constants
   const int size = SH::Terms(order);

   // Load the BRDF
   MerlBRDF brdf;
   if(! brdf.read_brdf(filename)) {
      std::cerr << "Failed: unable to load the MERL brdf" << std::endl;
      return 1;
   }

   // Values
   std::vector<Eigen::MatrixXf> cijs(3, Eigen::MatrixXf::Zero(size, size));
   const auto dirs = SamplingFibonacci<Vector>(N);
   /*
   for(unsigned int i=0; i<dirs.size(); ++i) {
      const auto& wo = dirs[i];

      // Skip below the horizon configuration
      if(wo.z < 0.0) continue;
      const auto ylmo = SH::FastBasis(wo, order);

      for(const auto& wi : dirs) {
         // Skip below the horizon configuration
         if(wi.z < 0.0) continue;

         // Evaluate the BRDF value
         const auto rgb = brdf.value<Vector, Vector>(wi, wo);
         const auto ylmi = SH::FastBasis(wi, order);

         const Eigen::MatrixXf mat = ylmo * ylmi.transpose();
         cij0 += rgb[0]*mat;
         cij1 += rgb[1]*mat;
         cij2 += rgb[2]*mat;
      }
   }
   */
   const int nbthreads = std::thread::hardware_concurrency();
   std::vector<MerlProjectionThread*> threads;
   for(int k=0; k<nbthreads; ++k) {
      MerlProjectionThread* th = new MerlProjectionThread(&brdf, &dirs, order, k, nbthreads);
      threads.push_back(th);
   }

   for(MerlProjectionThread* th : threads) {
      th->join();
      cijs[0] += th->cijs[0];
      cijs[1] += th->cijs[1];
      cijs[2] += th->cijs[2];
      delete th;
   }
   const float factor = 16.0*M_PI*M_PI / float(N*N);
   cijs[0] *= factor;
   cijs[1] *= factor;
   cijs[2] *= factor;

   SaveMatrices("gold-paint.mats", cijs);

   // Print values
   std::ofstream file("test.txt", std::ios_base::trunc);
   const float thetai = -0.5f*M_PI * 30.f/90.f;
   const Vector wi(sin(thetai), 0, cos(thetai));
   const auto ylmi = SH::FastBasis(wi, order);
   Vector wo;
   for(int i=0; i<90; ++i) {
      const float theta = 0.5*M_PI * i / float(90);
      wo.x = sin(theta);
      wo.y = 0;
      wo.z = cos(theta);

      // Ref
      const auto RGB = brdf.value<Vector, Vector>(wi, wo);
      const float R = RGB[0];
      const float G = RGB[1];
      const float B = RGB[2];

      // SH expansion
      const auto ylmo = SH::FastBasis(wo, order);
      const Eigen::VectorXf rlm = cijs[0] * ylmo;
      const Eigen::VectorXf glm = cijs[1] * ylmo;
      const Eigen::VectorXf blm = cijs[2] * ylmo;
      const float r = ylmi.dot(rlm);
      const float g = ylmi.dot(glm);
      const float b = ylmi.dot(blm);

      file << theta << "\t" << r << "\t" << g << "\t" << b
                    << "\t" << R << "\t" << G << "\t" << B
                    << std::endl;
   }

   int nb_fails = 0;
   return nb_fails;
}

int main(int argc, char** argv) {

   int nb_fails = 0;

   // Load an example
   //nb_fails += TestMerlProjectionSlice("gold-paint.binary");
   nb_fails += TestMerlProjectionMatrix("gold-paint.binary");

   if(nb_fails > 0) {
      return EXIT_FAILURE;
   } else {
      return EXIT_SUCCESS;
   }
}
