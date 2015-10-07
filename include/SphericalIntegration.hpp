#pragma once

// Include Eigen
#include <Eigen/Core>

// Include STL
#include <cmath>

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
      W.col(n).segment(1, subsize) = (2*n+1) * W.col(n-1).segment(0, subsize);
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