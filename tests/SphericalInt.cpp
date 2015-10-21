// STL includes
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>

// Local includes
#include "SphericalIntegration.hpp"
#include "DirectionsSampling.hpp"
#include "SH.hpp"
#include "Tests.hpp"

// GLM include
#include <glm/glm.hpp>

/* Check the Lengendre expansion for the first elements in the matrix. This
 * test the init elements and the recursion pattern.
 */
int CheckLegendreExpansion(float Epsilon = 1.0E-5f) {

   Eigen::MatrixXf W = ZonalWeights(3);
   Eigen::MatrixXf R (3,3);

   // Exact values for the first three elements of the serie.
   const float factor = 1.0f/sqrt(4.0f*M_PI);
   R << 1.0f,       0.0f, -sqrt(5.0f/4.0f),
        0.0f, sqrt(3.0f),             0.0f,
        0.0f,       0.0f,  sqrt(45.0f/4.0f);
   R *= factor;

   if(! W.isApprox(R, Epsilon)) {
      std::cout << "Error, ZonalWeights are not correct:" << std::endl;
      std::cout << W << std::endl << std::endl;
      std::cout << R << std::endl;
      return 1;
   } else {
      return 0;
   }
}

/* Check if the ZH expansion and the power of cosines expansion do match against
 * a variety of direction.
 */
int CheckZHEqualsCosinePower(float Epsilon = 1.0E-5f) {

   int nb_fails = 0;

   // Values
   const glm::vec3 n(0,0,1);
   const glm::vec3 w = glm::normalize(glm::vec3(1,1,1));
   const float dotWN = glm::dot(w, n);

   // Number of orders
   const int lmax = 10;

   // Vector of cosine power
   auto CosPow = Eigen::VectorXf(lmax);
   for(int i=0; i<lmax; ++i) {
      CosPow[i] = pow(dotWN, i);
   }
   auto W = ZonalWeights(lmax);
   auto CosZh = (CosPow.transpose() * W).transpose();

   // SH Evaluation for w.
   // Zonal coefficients represent y_l (w · n).
   auto ShBasis = SHEvalFast<glm::vec3>(w, lmax);
   auto ZhBasis = Eigen::VectorXf(lmax);
   for(int l=0; l<lmax; ++l) {
      ZhBasis[l] = ShBasis(SHIndex(l, 0));
   }

   if(! CosZh.isApprox(ZhBasis, Epsilon)) {
      std::cout << "Error: the convertion from power of cosine to ZH failed."
                << std::endl
                << "CosZh   = [" << CosZh.transpose()   << "]" << std::endl
                << "ZhBasis = [" << ZhBasis.transpose() << "]" << std::endl
                << std::endl;
      ++nb_fails;
   }

   return nb_fails;
}

int CheckZHExpansion(float Epsilon = 1.0E-5f) {

   int nb_fails = 0;
   int order    = 5;

   // Check if the ZH expansion correspond to the (l,0) elements in the
   // SH expansion.
   Vector n(0,0,1);

   // Test with respecto to the same direction. This should return the scaling
   // coeffcient to ensure that normalization per band is correct.
   Vector w = Vector::Normalize(Vector(0,0,1));
   std::cout << "Testing ZH == SH for Vector w = " << w << std::endl;
   auto zlm = RotatedZH(n, w, order);
   auto ylm = SHEvalFast(w, order);
   for(int l=0; l<order; ++l) {
      if(std::abs(zlm[l] - ylm[SHIndex(l, 0)]) > Epsilon) {
         ++nb_fails;
         std::cout << "Index " << l << " ZH coeff " << zlm[l]
                   << " != " << ylm[SHIndex(l, 0)] << std::endl;
      }
   }

   // Test with respect to an orthogonal vector, the resulting coeffcients
   // should all be zero.
   w = Vector::Normalize(Vector(1,0,0));
   std::cout << "Testing ZH == SH for Vector w = " << w << std::endl;
   zlm = RotatedZH(n, w, order);
   ylm = SHEvalFast(w, order);
   for(int l=0; l<order; ++l) {
      if(std::abs(zlm[l] - ylm[SHIndex(l, 0)]) > Epsilon) {
         ++nb_fails;
         std::cout << "Index " << l << " ZH coeff " << zlm[l]
                   << " != " << ylm[SHIndex(l, 0)] << std::endl;
      }
   }

   // Test with respect to to a non aligned vector
   w = Vector::Normalize(Vector(1,1,1));
   std::cout << "Testing ZH == SH for Vector w = " << w << std::endl;
   zlm = RotatedZH(n, w, order);
   ylm = SHEvalFast(w, order);
   for(int l=0; l<order; ++l) {
      if(std::abs(zlm[l] - ylm[SHIndex(l, 0)]) > Epsilon) {
         ++nb_fails;
         std::cout << "Index " << l << " ZH coeff " << zlm[l]
                   << " != " << ylm[SHIndex(l, 0)] << std::endl;
      }
   }

   // Test with respect to to a non aligned vector for low order
   w = Vector::Normalize(Vector(1,1,1));
   order = 1;
   std::cout << "Testing ZH == SH for Vector w = " << w << " with order = " << order << std::endl;
   zlm = RotatedZH(n, w, order);
   ylm = SHEvalFast(w, order);
   for(int l=0; l<order; ++l) {
      if(std::abs(zlm[l] - ylm[SHIndex(l, 0)]) > Epsilon) {
         ++nb_fails;
         std::cout << "Index " << l << " ZH coeff " << zlm[l]
                   << " != " << ylm[SHIndex(l, 0)] << std::endl;
      }
   }



   std::vector<Vector> directions;
   directions.push_back(glm::vec3(0,0,1));
   directions.push_back(glm::vec3(1,0,0));
   directions.push_back(glm::vec3(0,1,0));

   zlm = ZHEvalFast<Vector>(directions, directions[0]);

   if(std::abs(zlm[SHIndex(1,0)]) > Epsilon) {
      ++nb_fails;
   }
   if(std::abs(zlm[SHIndex(1,1)]) > Epsilon) {
      ++nb_fails;
   }

   std::cout << std::endl;
   return nb_fails;
}


std::mt19937 gen(0);
std::uniform_real_distribution<float> dist(0.0,1.0);

int CheckZHDecomposition(const std::vector<Vector>& directions,
                         float Epsilon = 1.0E-5f) {

   struct SH {
      // Define the Vector type
      typedef glm::vec3 Vector;

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

   int nb_fails = 0;
   int order    = (directions.size()-1) / 2;

   auto Y = ZonalExpansion<SH, Vector>(directions);
   auto A = Y.inverse();

   // Test if the matrix is full Rank
   Eigen::FullPivLU<Eigen::MatrixXf> rankComputer(Y);
   const int rank = rankComputer.rank();
   if(rank != SHTerms(order)) {
      std::cout <<std::setprecision(5) << std::fixed;
      for(int i=0; i<directions.size(); ++i) {
         std::cout << "w_" << i << " = " << directions[i] << std::endl;
      }
      std::cout << "Rank of matrix Y = " << rank
                << " / " << SHTerms(order) << std::endl;
   }

   // Generate a random SH vector and evaluate it with SH and ZH
   Eigen::VectorXf clm = Eigen::VectorXf::Zero(SHTerms(order));
   clm[SHIndex(order/2, 0)] = 1.0;
   auto w = Vector(0,0,1);

   // Evaluate the SH function
   const auto ylm = SHEvalFast<Vector>(w, order);
   //std::cout << clm.dot(ylm) << std::endl;

   // Evaluate the ZH function
   auto zlm = ZHEvalFast<Vector>(directions, w);
   //std::cout << clm.dot( A * zlm ) << std::endl;

   const auto ylmFromZml = Y.transpose()*zlm;
   for(int i=0; i<ylm.size(); ++i) {
      if(std::abs(ylm[i] - ylmFromZml[i]) > Epsilon) {
         ++nb_fails;
      }
   }

   if(nb_fails > 0) {
      std::cout <<std::setprecision(5) << std::fixed;
      std::cout << "Error, the conversion Zlm => Ylm failed" << std::endl;
      std::cout << std::endl;

      std::cout << "Set of directions:" << std::endl;
      for(int i=0; i<directions.size(); ++i) {
         std::cout << "w_" << i << " = " << directions[i] << std::endl;
      }
      std::cout << std::endl;

      std::cout << "ZLm    = " << zlm.transpose() << std::endl;
      std::cout << "Ylm    = " << ylm.transpose() << std::endl;
      std::cout << "Y'*Zlm = " << (Y.transpose()*zlm).transpose() << std::endl;
      std::cout << "Y *Zlm = " << (Y*zlm).transpose() << std::endl;
      std::cout << std::endl;

      std::cout << "coeffs = " << clm.transpose() << std::endl;
      std::cout << std::endl;

      std::cout << "Matrix Y:" << std::endl;
      std::cout << Y << std::endl;
   }

   return nb_fails;
}

int main(int argc, char** argv) {

   int nb_fails = 0;

   /* Check the basic elements of the ZH decomposition */
   nb_fails += CheckLegendreExpansion();

   /* Check if Zonal Harmonics expansion and the power of cosine
    * expansion match */
   nb_fails += CheckZHEqualsCosinePower();

   /* Check the Zonal Hamonics expansion provide coherent vectors */
   nb_fails += CheckZHExpansion();

   /* Check if the Zonal Harmonics expansion and the SH basis match */

   // Using random directions
   int order = 2;
   std::vector<Vector> directions;
   directions = SamplingBlueNoise<Vector>(2*order+1);
   nb_fails += CheckZHDecomposition(directions);

   // Using the canonical frame
   directions.clear();
   directions.push_back(glm::normalize(glm::vec3(0,0,1)));
   directions.push_back(glm::normalize(glm::vec3(0,1,0)));
   directions.push_back(glm::normalize(glm::vec3(1,0,0)));
   nb_fails += CheckZHDecomposition(directions);

   if(nb_fails == 0) {
      return EXIT_SUCCESS;
   } else {
      return EXIT_FAILURE;
   }
}
