// STL includes
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <random>
#include <utility>
#include <thread>

// Local includes
#include "Merl.hpp"
#include <tests/Tests.hpp>
#include <tests/SH.hpp>
#include <include/Utils.hpp>
#include <include/SphericalHarmonics.hpp>
#include <include/DirectionsSampling.hpp>
#include <include/SphericalIntegration.hpp>

// GLM include
#include <glm/glm.hpp>

// Include Eigen
#include <Eigen/Core>

struct MerlProjectionThread : public std::thread {

   int order;
   std::vector<Eigen::MatrixXf> cijs;

   MerlProjectionThread(const MerlBRDF* brdf,
                        const std::vector<Vector>* ws,
                        int order, int skip, int nthread) :
      std::thread(&MerlProjectionThread::run, this, brdf, ws, skip, nthread),
      order(order),
      cijs(6, Eigen::MatrixXf::Zero(SH::Terms(order), SH::Terms(order)))
   {}

   /* Pre-convolved the Zonal basis to perform smooth evaluation.
    */
   static void ApplyZonalFilter(Eigen::VectorXf& clm) {
      const auto bmax = floor(sqrt(clm.size())-1);
      for(unsigned int i=0; i<clm.size(); ++i) {
         const auto b = floor(sqrt(i));
         const auto f = exp(- pow(float(b)/float(bmax), 2));
         clm[i] *= f;
      }
   }

   void run(const MerlBRDF* brdf,
            const std::vector<Vector>* dirs,
            int skip, int nthread) {

      const int size = SH::Terms(order);
      Eigen::VectorXf ylmo(size);
      Eigen::VectorXf ylmi(size);
      for(unsigned int i=skip; i<dirs->size(); i+=nthread) {
         const Vector& wo = (*dirs)[i];

         if(skip == 0) {
            std::cout << "Progress: " << i << " / " << dirs->size() << "     \r";
            std::cout.flush();
         }

         // Skip below the horizon configuration
         if(wo.z < 0.0) continue;
         SH::FastBasis(wo, order, ylmo);

         for(unsigned int j=0; j<dirs->size(); ++j) {
            const Vector& wi = (*dirs)[j];
            // Skip below the horizon configuration
            if(wi.z < 0.0) continue;

            // Evaluate the BRDF value
            const auto rgb = brdf->value<Vector, Vector>(wi, wo);
            SH::FastBasis(wi, order, ylmi);

            // Apply filtering
            //ApplyZonalFilter(ylmo);
            //ApplyZonalFilter(ylmi);

            Eigen::MatrixXf mat = ylmo * ylmi.transpose();
#ifndef SYMMETRIZE
            mat = 0.5f*(mat + mat.transpose());
#endif
            cijs[0] += rgb[0] * mat;
            cijs[1] += rgb[1] * mat;
            cijs[2] += rgb[2] * mat;
            // Note: Here the correct weighting should be with respect to wi.z
            // but I use wo.z since it allows to reduce the ringing drastically.
#ifdef LOOKS_BETTER
            cijs[3] += rgb[0] * wo.z * mat;
            cijs[4] += rgb[1] * wo.z * mat;
            cijs[5] += rgb[2] * wo.z * mat;
#else // CORRECT
            cijs[3] += rgb[0] * wi.z * mat;
            cijs[4] += rgb[1] * wi.z * mat;
            cijs[5] += rgb[2] * wi.z * mat;
#endif
         }
      }
   }
};

int MerlProjectionMatrix(const std::string& filename,
                         int order = 15, int N = 1000) {

   // Constants
   const int size = SH::Terms(order);

   // Load the BRDF
   MerlBRDF brdf;
   if(! brdf.read_brdf(filename)) {
      std::cerr << "Failed: unable to load the MERL brdf" << std::endl;
      return 1;
   }

   const auto k = filename.rfind('.');
   std::string ofilename = filename;
   ofilename.replace(k, std::string::npos, ".mats");
   std::cout << "Will output to \"" << ofilename << "\"" << std::endl;

   // Values
   std::vector<Eigen::MatrixXf> cijs(6, Eigen::MatrixXf::Zero(size, size));
   const auto dirs = SamplingFibonacci<Vector>(N);

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
      cijs[3] += th->cijs[3];
      cijs[4] += th->cijs[4];
      cijs[5] += th->cijs[5];
      delete th;
   }
   const float factor = 16.0*M_PI*M_PI / float(N*N);
   cijs[0] *= factor;
   cijs[1] *= factor;
   cijs[2] *= factor;
   cijs[3] *= factor;
   cijs[4] *= factor;
   cijs[5] *= factor;

   SaveMatrices(ofilename, cijs);

   // Print values
   std::string gfilename = filename;
   gfilename.replace(k, std::string::npos, ".gnuplot");
   std::ofstream file(gfilename.c_str(), std::ios_base::trunc);
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

bool parseArguments(int argc, char** argv, std::string& filename,
                    int& order, int& nb) {

   // Loop over all the different elements of the command line and search for
   // some patterns
   for(int k=0; k<argc; ++k) {
      if(argv[k] == std::string("-h") || argv[k] == std::string("--help")) {
         std::cerr << "Usage: merl2sh [options] filename.binary" << std::endl;
         return false;
      }
      if((argv[k] == std::string( "-o") || argv[k] == std::string("--order")) && k+1<argc) {
         order = std::atoi(argv[k+1]);
      }
      if((argv[k] == std::string( "-n") || argv[k] == std::string("--nb")) && k+1<argc) {
         nb = std::atoi(argv[k+1]);
      }
   }

   // The filename to convert is the last argument of the command line.
   if(argc > 1) {
      filename = argv[argc-1];
      return true;
   } else {
      return false;
   }
}

int main(int argc, char** argv) {

   int nb_fails = 0;
   std::string filename;
   int order = 3;
   int nb = 1000;
   if(! parseArguments(argc, argv, filename, order, nb)) {
      return EXIT_SUCCESS;
   }

   // Load an example
   nb_fails += MerlProjectionMatrix(argv[argc-1], order, nb);

   if(nb_fails > 0) {
      return EXIT_FAILURE;
   } else {
      return EXIT_SUCCESS;
   }
}
