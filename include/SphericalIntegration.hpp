#pragma once

// Include Eigen
#include <Eigen/Core>
#include <Eigen/LU>
#ifdef USE_SPARSE_EIGEN
#include <Eigen/Sparse>
typedef Eigen::SparseMatrix<float> MatrixType;
#else
typedef Eigen::MatrixXf MatrixType;
#endif

// Include STL
#include <cmath>
#include <vector>

// Local include
#include "AxialMoments.hpp"

/* _Zonal Weigths_
 *
 * Compute the matrix of weights w_{l,i} that converts dot powers (wd, w)^i
 * to a Zonal Hamonics. More precisely:
 *
 *     y_{l, 0}(w · wd) = ∑_i w_{l,i} (w · wd)^i, V l in [0 .. order]
 *
 */
inline Eigen::MatrixXf ZonalWeights(int order) {

   // W is the matrix of the weights. It is build recursively using the Legendre
   // polynomials. The trick here is that Legendre polynomials are simply the
   // shift in order of a previous pol. summed to another previous pol.
   Eigen::MatrixXf W = Eigen::MatrixXf::Zero(order, order);
   W(0,0) = 1.0f;
   W(1,1) = 1.0f;
   for(int n=2; n<order; ++n) {
      const int subsize = n;
      W.row(n).segment(1, subsize) = (2*n-1) * W.row(n-1).segment(0, subsize);
      W.row(n) -= (n-1) * W.row(n-2);
      W.row(n) /= n;
   }

   // Scale each columns by the corresponding √(2l+1 / 4π) factor.
   const float factor = 1.0f/sqrt(4.0f*M_PI);
   for(int n=0; n<W.rows(); ++n) {
      W.row(n) = W.row(n) * sqrt(2*n+1) * factor;
   }

   return W;
}

/* _Vector Group Zonal Weigths_
 *
 * TODO This matrix should do the mixing between the ordered power of cosines
 * vector and the shuffled vector of ZH coefficients.
 */
inline Eigen::MatrixXf ZonalWeights(const std::vector<Vector>& directions) {

   const int dsize = directions.size();
   const int order = (dsize-1) / 2 + 1;
   const int mrows = order*order;

   Eigen::MatrixXf result = Eigen::MatrixXf::Zero(mrows, dsize*order);

   const auto ZW = ZonalWeights(order);

   // Each vector fills a given set of column entries with decreasing
   // number to do the swapping. For example, the 0th vector will fill
   // entries from order 0 to max_order but the 1srt vector will fill
   // entries from 1 to max_order.
   for(int i=0; i<dsize; ++i) {
      for(int j=0; j<order; ++j) {
         if(i >= 2*j+1) {
            continue;
         }

         const int shift_rows = j*j + i;
         const int shift_cols = order*i;

         result.block(shift_rows, shift_cols, 1, order) = ZW.row(j);
      }
   }
   return result;
}

/* _Zonal Normalization_
 *
 * This little code compute the zonal normalization for spherical harmonics
 * basis using the √(2l+1 / 4π) factor. It can later by applied to a zonal
 * vector using Eigen fast product: a.cwiseProduct(b)
 */
Eigen::VectorXf ZonalNormalization(int order) {
   Eigen::VectorXf res = Eigen::VectorXf(order);
   const float f = 1.0f/sqrt(4.0f*M_PI);
   for(int i=0; i<order; ++i) {
      res[i] = sqrt(2*i+1)*f;
   }
   return res;
}

/* _Rotated ZH_
 *
 * Evaluate the rotated ZH, with respect to principal direction `d` in
 * direction `w` for order in [0..order].
 *
 *   z_i = P_i( d · w ),  where P_i is Legendre polynomial
 */
template<class Vector>
Eigen::VectorXf RotatedZH(const Vector& d, const Vector& w, int order) {

   Eigen::VectorXf res = Eigen::VectorXf::Zero(order);
   const float z = Vector::Dot(d, w);

   res[0] = 1.0f;
   if(order == 1) {
      return res /sqrt(4.0f*M_PI);
   }

   res[1] = z;
   if(order == 2) {
      res[1] *= sqrt(3.0f);
      return res / sqrt(4.0f*M_PI);
   }

   // Using Bonnet recurrence formula
   for(int i=2; i<order; ++i) {
      const int i2 = i-1;
      res[i]  = (2*(i2)+1)*z*res[i2] - (i2)*res[i-2];
      res[i] /= i;
   }

   // Apply zonal weights
   const auto f = ZonalNormalization(order);
   res = res.cwiseProduct(f);
   return res;
}

/* _ZH Fast Evaluation_
 *
 * Evaluate the Zonal Harmonics vector for all `directions` while respecting
 * a lobe sharing order. This means that the output vector will be composed as:
 *
 *     v = [ z_{0,0}, z_{1,0}, z_{1,1}, z_{1,2}, ...]
 *
 * where z_{i,j} = Ylm_{i, 0}(w_j).
 */
template<class Vector>
Eigen::VectorXf ZHEvalFast(const std::vector<Vector>& dirs, const Vector& w) {

   // Get the number of elements to compute
   const int dsize = dirs.size();
   const int order = (dsize-1) / 2 + 1;
   const int vsize = order*order;

   Eigen::VectorXf v(vsize);

   // The loop is first over the order term then over the set of directions
   // starting from the 0 index that is reused for all bands.
   float ylm = 0.0;
   for(int i=0; i<dsize; ++i) {

      const Vector& n   = dirs[i];
      const auto    ylm = RotatedZH<Vector>(n, w, order);

      for(int l=0; l<order; ++l) {
         if(i >= 2*l+1) {
            continue;
         }
         v[l*l + i] = ylm[l];
      }
   }

   return v;
}


/* _Zonal Expansion_
 *
 * Expands a Spherical Harmonics vector into a Zonal Harmonics matrix transform.
 * Each band of spherical harmonics is decomposed into a set of Zonal Hamonics
 * with specific directions `directions`.
 *
 * The conversion matrix A is computed as the inverse of the evaluation of the
 * SH basis a the particular directions.
 *
 * The vector of directions must be of size `2*order+1`, where `order` is the
 * max order of the SH expansion.
 *
 * The template class SH must permit to access its basis elements, the y_{l,m}
 * as the static function `SH::FastBasis(const Vector& w, int order)`.
 */
template<class SH, class Vector>
MatrixType ZonalExpansion(const std::vector<Vector>& directions) {

   // Get the current band. Here I use a shifted order number. The integer
   // order is actually `order+1` to compute the number of rows and iterate
   // over it. Later in the code I evaluate the correct order to get the
   // ylm from SH::FastBasis.
   const int dsize = directions.size();
   const int order = (dsize-1) / 2 + 1;
   const int mrows = order*order;
   assert(order >= 0);

   const auto zhNorm = ZonalNormalization(order);

#ifdef USE_SPARSE_EIGEN
   MatrixType Y(mrows, mrows);
   std::vector<Eigen::Triplet<float>> triplets;
#else
   MatrixType Y = Eigen::MatrixXf::Zero(mrows, mrows);
#endif
   for(int i=0; i<dsize; ++i) {

      // Get the vector associated to the current row
      const Vector& w = directions[i];

      // Evaluate all the Y_{l,m} for the current vector
      auto ylm = SH::FastBasis(w, order-1);

      // Complete all the submatrices starting from column `i` using order `j`
      // size for the ZH element.
      for(int j=0; j<order; ++j) {
         if(i >= 2*j+1) {
            continue;
         }

         const int  shift = j*j;
         const int  vsize = 2*j+1;
#ifdef USE_SPARSE_EIGEN
         for(int k=0; k<vsize; ++k) {
            const float& v = ylm[shift+k];
            triplets.push_back(Eigen::Triplet<float>(shift+i, shift+k, v));
         }
         Y.setFromTriplets(triplets.begin(), triplets.end());
#else
         Y.block(shift+i, shift, 1, vsize) = ylm.segment(shift, vsize).transpose() / zhNorm[j];
#endif
      }
   }

   return Y;
}

/* _Compute the Inverse of Matrix Y_
 *
 * Since the matrix resulting from ZonalExpansion is block diagonal, its
 * inverse is trivial to compute. We must simply take the inverse of each
 * block, in place.
 *
 * TODO: Make the sparse version.
 */
inline MatrixType computeInverse(const MatrixType& Y) {
   const int nrows = Y.rows();
   const int order = sqrt(nrows);

#ifdef USE_SPARSE_EIGEN
   MatrixType A(mrows, nrows);
   std::vector<Eigen::Triplet<float>> triplets;
#else
   MatrixType A = Eigen::MatrixXf::Zero(nrows, nrows);
#endif

   for(int j=0; j<order; ++j) {
      const int shift = j*j;
      const int size  = 2*j+1;

#ifdef USE_SPARSE_EIGEN
      assert(false);
#else
      const auto& block = Y.block(shift, shift, size, size);

      // Check if the block is Zero or not in debug mode
      if(block.isZero()) {
         std::cout << "Block of order " << j << " is zero" << std::endl;
         std::cout << block << std::endl << std::endl;
      }
      //assert(!block.isZero());

      A.block(shift, shift, size, size) = block.inverse();
#endif
   }
   return A;
}

/* _Compute the SH Integral over a Spherial Triangle_
 *
 * This function is provided as an example of how to use the different
 * components of this package. It is probably much faster to precompute the
 * product `Prod` of the ZonalWeights and the Zonal to SH conversion matrix.
 */
template<class Triangle, class Vector, class SH>
inline float computeSHIntegral(const Eigen::VectorXf& clm,
                               const std::vector<Vector>& basis,
                               const Triangle& triangle) {

   // Get the Zonal weights matrix and the Zlm -> Ylm conversion matrix
   // and compute the product of the two: `Prod = A x Zw`.
   const auto ZW = ZonalWeights(basis);
   const auto Y  = ZonalExpansion<SH, Vector>(basis);
   const auto A  = computeInverse(Y);

   const auto Prod = A*ZW;

   // Analytical evaluation of the integral of power of cosines for
   // the different basis elements up to the order defined by the
   // number of elements in the basis
   const auto moments = AxialMoments<Triangle, Vector>(triangle, basis);

   return clm.dot(Prod * moments);

}
