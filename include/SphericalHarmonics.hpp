#pragma once

// Eigen includes
#include <Eigen/Core>
#include <Eigen/LU>

// STL includes
#include <vector>

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

   for(int i=0; i<basis.size(); ++i) {

      const Vector& w = basis[i];

      auto ylm = SH::FastBasis(w, order);
      Ylm.row(i) = ylm;

      flm[i] = f(w);
   }

   return Ylm.inverse() * flm;
}