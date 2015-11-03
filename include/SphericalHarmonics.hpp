#pragma once

// Eigen includes
#include <Eigen/Core>
#include <Eigen/LU>

// STL includes
#include <vector>
#include <random>

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
Eigen::VectorXf ProjectToShMC(const Functor& f, int order, int M=100000) {

   std::mt19937 gen(0);
   std::uniform_real_distribution<float> dist(0.0,1.0);

   Eigen::VectorXf shCoeffs((order+1)*(order+1));
   for(int i=0; i<M; ++i) {

      // Sample the cosine of the elevation
      Vector w;
      w.z = 2.0*dist(gen) - 1.0;
      const float z2 = w.z*w.z;

      // Sample the azimuth
      const float phi = 2.0*M_PI*dist(gen);
      w.x = sqrt(1.0f-z2) * cos(phi);
      w.y = sqrt(1.0f-z2) * sin(phi);

      // Evaluate the function and the basis vector
      shCoeffs += f(w) * SH::FastBasis(w, order);
   }
   shCoeffs *= 4.0*M_PI / float(M);

   return shCoeffs;
}
