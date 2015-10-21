#pragma once

// Include Eigen
#include <Eigen/Core>
#include <Eigen/LU>

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
Eigen::MatrixXf ZonalWeights(int order) {

   // W is the matrix of the weights. It is build recursively using the Legendre
   // polynomials. The trick here is that Legendre polynomials are simply the
   // shift in order of a previous pol. summed to another previous pol.
   Eigen::MatrixXf W = Eigen::MatrixXf::Zero(order, order);
   W(0,0) = 1.0f;
   W(1,1) = 1.0f;
   for(int n=2; n<order; ++n) {
      const int subsize = n;
      W.col(n).segment(1, subsize) = (2*n-1) * W.col(n-1).segment(0, subsize);
      W.col(n) -= (n-1) * W.col(n-2);
      W.col(n) /= n;
   }

   // Scale each columns by the corresponding √(2l+1 / 4π) factor.
   const float factor = 1.0f/sqrt(4.0f*M_PI);
   for(int n=0; n<W.cols(); ++n) {
      W.col(n) = W.col(n) * sqrt(2*n+1) * factor;
   }

   return W;
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
Eigen::MatrixXf ZonalExpansion(const std::vector<Vector>& directions) {

   // Get the current band
   const int dsize = directions.size();
   const int order = (dsize-1) / 2 + 1;
   const int mrows = order*order;
   assert(order >= 0);

   const auto zhNorm = ZonalNormalization(order);

   Eigen::MatrixXf Y = Eigen::MatrixXf::Zero(mrows, mrows);
   for(int i=0; i<dsize; ++i) {

      // Get the vector associated to the current row
      const Vector& w = directions[i];

      // Evaluate all the Y_{l,m} for the current vector
      auto ylm = SH::FastBasis(w, order);

      // Complete all the submatrices starting from column `i` using order `j`
      // size for the ZH element.
      for(int j=0; j<order; ++j) {
         if(i >= 2*j+1) {
            continue;
         }

         const int  shift = j*j;
         const int  vsize = 2*j+1;
         Y.block(shift+i, shift, 1, vsize) = ylm.segment(shift, vsize).transpose() / zhNorm[j];
      }
   }

   return Y;
}
