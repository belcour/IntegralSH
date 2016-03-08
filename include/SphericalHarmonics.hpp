#pragma once

// Eigen includes
#include <Eigen/Core>
#include <Eigen/LU>

// STL includes
#include <vector>
#include <random>
#ifdef USE_THREADS
#include <thread>
#endif

// Local include
#include "DirectionsSampling.hpp"

/* This header provide multiple ways to project spherical function to Spherical
 * Harmonics vectors.
 *
 *    + 'ProjectToSH' return the SH coefficient vector when the Functor to be
 *    projected is bandlimited to the max order of SH. Warning: do not use with
 *    highly varying functions! This projection will probably behave very
 *    badly. In such case, it is advised to use the 'ProjectToShMC' method.
 *
 *    + 'ProjectToShMC' return the SH coefficient vector by integrating <f·Ylm>
 *    using Monte Carlo integration. It is possible to choose the type of
 *    random or quasi- random sequence for the integration.
 *
 *    + 'TripleTensorProduct' return the premultiplied triple tensor product:
 *    the integral of <Ylm x Ylm x Ylm> multiplied by the SH coefficients 'clm'
 *    of 'f'.  The integral is computed using Monte Carlo, as 'ProjectToShMC'
 *    does.
 *
 * TODO: Template the direction sequence.
 */

#define USE_FIBONACCI_SEQ
//#define USE_BLUENOISE_SEQ

/* From a set of `basis` vector directions, and a spherical function `Functor`,
 * generate the Spherical Harmonics projection of the `Functor`. This algorithm
  * works as follows:
 *
 *  1. Evaluate the matrix Ylm(w_i) for each SH index and each direction
 *  2. Evaluate the vector of the functor [f(w_0), ..., f(w_n)]
 *  3. Return the product Ylm(w_i)^{-1} [f(w_0), ..., f(w_n)]
 *
 *  Requierement: `basis` must have the dimension of the output SH vector.
 *                `f` must be real valued. Higher dimension functor are not
 *                handled yet.
 */
template<class Functor, class Vector, class SH>
inline Eigen::VectorXf ProjectToSH(const Functor& f,
                                   const std::vector<Vector>& basis) {

   // Get the number of elements to compute
   const int dsize = basis.size();
   const int order = sqrt(dsize)-1;

   Eigen::MatrixXf Ylm(dsize, dsize);
   Eigen::VectorXf flm(dsize);

   for(unsigned int i=0; i<basis.size(); ++i) {

      const Vector& w = basis[i];

      auto ylm = SH::FastBasis(w, order);
      Ylm.row(i) = ylm;

      flm[i] = f(w);
   }

   return Ylm.inverse() * flm;
}

/* Project a spherical function `f` onto Spherical Harmonics coefficients up
 * to order `order` using Monte-Carlo integration.
 */
template<class Functor, class Vector, class SH>
inline Eigen::VectorXf ProjectToShMC(const Functor& f, int order, int M=100000) {

   std::mt19937 gen(0);
   std::uniform_real_distribution<float> dist(0.0,1.0);

   Eigen::VectorXf shCoeffs((order+1)*(order+1));
#if   defined(USE_FIBONACCI_SEQ)
   const std::vector<Vector> directions = SamplingFibonacci<Vector>(M);
#elif defined(USE_BLUENOISE_SEQ)
   const std::vector<Vector> directions = SamplingBlueNoise<Vector>(M);
#else
   const std::vector<Vector> directions = SamplingRandom<Vector>(M);
#endif
   for(auto& w : directions) {
      // Evaluate the function and the basis vector
      shCoeffs += f(w) * SH::FastBasis(w, order);
   }
   shCoeffs *= 4.0*M_PI / float(M);

   return shCoeffs;
}

/* Compute the triple tensor product \int Ylm * Ylm * Ylm
 *
 * ## Arguments:
 *
 *    + 'ylm' is the input spherical function SH coeffcients. They will be
 *      prefactored to the triple product tensor to build the matrix.
 *
 *    + 'truncated' allows to export either the truncated matrix (up to order
 *      'n', where 'n' is the order the input SH coeffs 'ylm') or the full matrix
 *      that is order '2n-1'.
 */
template<class SH, class Vector>
inline Eigen::MatrixXf TripleTensorProduct(const Eigen::VectorXf& ylm,
                                           bool truncated=true,
                                           int nDirections=100000) {

   // Compute the max order
   const int vsize = ylm.size();
   const int order = (truncated) ? sqrt(vsize)-1 : 2*sqrt(vsize)-2;
   const int msize = (truncated) ? vsize         : SH::Terms(order);

   Eigen::MatrixXf res = Eigen::MatrixXf::Zero(msize, msize);
   Eigen::VectorXf clm(msize);

   // Take a uniformly distributed point sequence and integrate the triple tensor
   // for each SH band
#if   defined(USE_FIBONACCI_SEQ)
   const std::vector<Vector> directions = SamplingFibonacci<Vector>(nDirections);
#elif defined(USE_BLUENOISE_SEQ)
   const std::vector<Vector> directions = SamplingBlueNoise<Vector>(nDirections);
#else
   const std::vector<Vector> directions = SamplingRandom<Vector>(nDirections);
#endif
   for(auto& w : directions) {
      SH::FastBasis(w, order, clm);
      res += clm.segment(0, vsize).dot(ylm) * clm * clm.transpose();
   }

   res *= 4.0f * M_PI / float(nDirections);
   return res;
}

/* Compute the triple tensor product \int Ylm * Ylm * Ylm for a bunch of
 * function projected on Spherical Harmonics. This method is specially
 * interesting to construct product of (R,G,B) component where each component
 * is stored in a separate vector.
 *
 * ## Arguments:
 *
 *    + 'ylm' is the input spherical function SH coeffcients. They will be
 *      prefactored to the triple product tensor to build the matrix.
 *
 *    + 'truncated' allows to export either the truncated matrix (up to order
 *      'n', where 'n' is the order the input SH coeffs 'ylm') or the full matrix
 *      that is order '2n-1'.
 */
template<class SH, class Vector>
inline std::vector<Eigen::MatrixXf> TripleTensorProduct(
                                       const std::vector<Eigen::VectorXf>& ylms,
                                       bool truncated=true,
                                       int nDirections=100000) {

#ifdef USE_THREADS
   struct TTPThread : public std::thread {

      std::vector<Eigen::MatrixXf> res;

      TTPThread(const std::vector<Vector>& dirs, unsigned int start, unsigned int end,
                const std::vector<Eigen::VectorXf>& ylms, int order) :
         std::thread(&TTPThread::run, this, dirs, start, end, ylms, order) {}

      void run(const std::vector<Vector>& dirs, unsigned int start, unsigned int end,
               const std::vector<Eigen::VectorXf>& ylms, int order) {
         const int vsize = ylms[0].size();
         const float fact = 4.0f * M_PI / float(dirs.size());

         const int msize = SH::Terms(order);
         Eigen::MatrixXf mat(msize, msize);
         Eigen::VectorXf clm(msize);

         res.reserve(3);
         res.push_back(Eigen::MatrixXf::Zero(msize, msize));
         res.push_back(Eigen::MatrixXf::Zero(msize, msize));
         res.push_back(Eigen::MatrixXf::Zero(msize, msize));

         for(unsigned int k=start; k<end; ++k) {
            // Get the vector
            const Vector& w = dirs[k];

            // Construct the matrix
            SH::FastBasis(w, order, clm);
            mat = clm * clm.transpose();

            // For each SH vector apply the weight to the matrix and sum it
            for(unsigned int i=0; i<ylms.size(); ++i) {
               res[i] += fact * clm.segment(0, vsize).dot(ylms[i]) * mat;
            }
         }
      }
   };
#endif

   // Compute the max order
   const int vsize = ylms[0].size();
   const int order = (truncated) ? sqrt(vsize)-1 : 2*sqrt(vsize)-2;
   const int msize = (truncated) ? vsize         : SH::Terms(order);

   std::vector<Eigen::MatrixXf> res(ylms.size(), Eigen::MatrixXf::Zero(msize, msize));

   // Take a uniformly distributed point sequence and integrate the triple tensor
   // for each SH band
#if   defined(USE_FIBONACCI_SEQ)
   const std::vector<Vector> directions = SamplingFibonacci<Vector>(nDirections);
#elif defined(USE_BLUENOISE_SEQ)
   const std::vector<Vector> directions = SamplingBlueNoise<Vector>(nDirections);
#else
   const std::vector<Vector> directions = SamplingRandom<Vector>(nDirections);
#endif

#ifdef USE_THREADS
   std::vector<TTPThread*> threads;
   const unsigned int nthreads = std::thread::hardware_concurrency();
   for(unsigned int i=0; i<nthreads; ++i) {
      const unsigned int block = nDirections / nthreads;
      const unsigned int start = i * block;
      const unsigned int end   = (i<nthreads-1) ? (i+1)*block : nDirections;
      threads.push_back(new TTPThread(directions, start, end, ylms, order));
   }

   for(TTPThread* thread : threads) {
      thread->join();

      for(unsigned int i=0; i<3; ++i) {
         res[i] += thread->res[i];
      }
   }
#else
   Eigen::VectorXf clm(msize);
   const float fact = 4.0f * M_PI / float(nDirections);
   for(auto& w : directions) {
      // Construct the matrix
      SH::FastBasis(w, order, clm);
      const auto matrix = clm * clm.transpose();

      // For each SH vector apply the weight to the matrix and sum it
      for(unsigned int i=0; i<ylms.size(); ++i) {
         res[i] += fact * clm.segment(0, vsize).dot(ylms[i]) * matrix ;
      }
   }
#endif
   return res;
}

template<class SH, class Vector>
Eigen::MatrixXf TripleTensorProductCos(int  order,
                                       int  nDirections=100000) {

   // Compute the max order
   const int msize = SH::Terms(order);

   Eigen::MatrixXf res = Eigen::MatrixXf::Zero(msize, msize);

   // Take a uniformly distributed point sequence and integrate the triple tensor
   // for each SH band
#if   defined(USE_FIBONACCI_SEQ)
   const std::vector<Vector> directions = SamplingFibonacci<Vector>(nDirections);
#elif defined(USE_BLUENOISE_SEQ)
   const std::vector<Vector> directions = SamplingBlueNoise<Vector>(nDirections);
#else
   const std::vector<Vector> directions = SamplingRandom<Vector>(nDirections);
#endif
   const float fact = 4.0f * M_PI / float(nDirections);
   for(auto& w : directions) {
      // Construct the matrix
      const Eigen::VectorXf clm = SH::FastBasis(w, order);
      const auto matrix = clm * clm.transpose();

      // For each SH vector apply the weight to the matrix and sum it
      if(w[2] > 0.0) {
         res += fact * w[2] * matrix;
      }
   }
   return res;
}
