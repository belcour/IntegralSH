// STL includes
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>

// Local includes
//#define USE_SPARSE_EIGEN
#include "Tests.hpp"
#include "SH.hpp"
#include "SphericalIntegration.hpp"
#include "DirectionsSampling.hpp"

// GLM include
#include <glm/glm.hpp>

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

/* Check the Lengendre expansion for the first elements in the matrix. This
 * test the init elements and the recursion pattern.
 */
int CheckLegendreExpansion(float Epsilon = 1.0E-3f) {

   Eigen::MatrixXf W = ZonalWeights(3);
   Eigen::MatrixXf R (3,3);

   // Exact values for the first three elements of the serie.
   const float factor = 1.0f/sqrt(4.0f*M_PI);
   R << 1.0f,       0.0f, -sqrt(5.0f/4.0f),
        0.0f, sqrt(3.0f),             0.0f,
        0.0f,       0.0f,  sqrt(45.0f/4.0f);
   R *= factor;

   if(! W.isApprox(R.transpose(), Epsilon)) {
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
int CheckZHEqualsCosinePower(float Epsilon = 1.0E-3f) {

   int nb_fails = 0;
   std::cout << "Testing if the ZH => Cosine power is correct." << std::endl;

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
   auto CosZh = (W * CosPow);

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

   if(nb_fails == 0) {
      std::cout << "Test passed!" << std::endl;
   }
   std::cout << std::endl;
   return nb_fails;
}

int CheckZHExpansion(float Epsilon = 1.0E-3f) {

   int nb_fails = 0;
   int order    = 15;

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
      if(! closeTo(zlm[l], ylm[SHIndex(l, 0)])) {
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
      if(! closeTo(zlm[l], ylm[SHIndex(l, 0)])) {
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
      if(! closeTo(zlm[l], ylm[SHIndex(l, 0)])) {
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
      if(! closeTo(zlm[l], ylm[SHIndex(l, 0)])) {
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

   if(! closeTo(zlm[SHIndex(1,0)], 0.0f)) {
      ++nb_fails;
   }
   if(! closeTo(zlm[SHIndex(1,1)], 0.0f)) {
      ++nb_fails;
   }

   if(nb_fails == 0) {
      std::cout << "Test passed!" << std::endl;
   }
   std::cout << std::endl;
   return nb_fails;
}


std::mt19937 gen(0);
std::uniform_real_distribution<float> dist(0.0,1.0);

int CheckZHDecomposition(const std::vector<Vector>& directions,
                         const std::vector<Vector>& queries,
                         float Epsilon = 1.0E-3f) {


   int nb_fails = 0;
   int order    = (directions.size()-1) / 2;

   std::cout << "Testing the conversion ZH <=> SH with an order " << order
             << " for a set of " << queries.size() << " directions" << std::endl;

   auto Y = ZonalExpansion<SH, Vector>(directions);
   if(!closeTo(Y(0,0), 1.0f, Epsilon)) {
      std::cout << "Error: first entry in the matrix is: "
                << Y(0,0) << " ≠ 1" << std::endl;
      ++nb_fails;
   }

   auto A = computeInverse(Y);

   // Test if the matrix is full Rank
   Eigen::FullPivLU<Eigen::MatrixXf> rankComputer(Y);
   const int rank = rankComputer.rank();
   if(rank != SHTerms(order)) {
      ++nb_fails;
      std::cout <<std::setprecision(5) << std::fixed;
      for(int i=0; i<directions.size(); ++i) {
         std::cout << "w_" << i << " = " << directions[i] << std::endl;
      }
      std::cout << "Rank of matrix Y = " << rank
                << " / " << SHTerms(order) << std::endl;
   }

   for(auto& w : queries) {
      // Evaluate the SH function
      const auto ylm = SHEvalFast<Vector>(w, order);

      // Evaluate the ZH function
      auto zlm = ZHEvalFast<Vector>(directions, w);

      bool ylmFromZmlFailed = false;
      bool zlmFromYmlFailed = false;

      const auto ylmFromZml = A*zlm;
      for(int i=0; i<ylm.size(); ++i) {
         //if(std::abs(ylm[i] - ylmFromZml[i]) > Epsilon) {
         if(! ylm.isApprox(ylmFromZml, Epsilon)) {
            ++nb_fails;
            ylmFromZmlFailed = true;
            std::cout << "Ylm => Zlm failed for coeff " << i << ": "
                      << ylm[i] << " ≠ " << ylmFromZml[i] << std::endl;
         }
      }

      const auto zlmFromYml = Y*ylm;
      for(int i=0; i<zlm.size(); ++i) {
         //if(std::abs(zlm[i] - zlmFromYml[i]) > Epsilon) {
         if(! zlm.isApprox(zlmFromYml, Epsilon)) {
            ++nb_fails;
            zlmFromYmlFailed = true;
            std::cout << "Zlm => Ylm failed for coeff " << i << ": "
                      << zlm[i] << " ≠ " << zlmFromYml[i] << std::endl;
         }
      }

      if(zlmFromYmlFailed || ylmFromZmlFailed) {
         std::cout <<std::setprecision(5) << std::fixed;
         std::cout << "Error, the conversion Ylm <=> Zlm failed for  direction w = "
                   << w << std::endl;
         std::cout << std::endl;

#ifdef FULL_DEBUG
         std::cout << "Set of directions:" << std::endl;
         for(int i=0; i<directions.size(); ++i) {
            std::cout << "w_" << i << " = " << directions[i] << std::endl;
         }
         std::cout << std::endl;

         std::cout << "ZLm    = " << zlm.transpose() << std::endl;
         std::cout << "Y *Ylm = " << (Y*ylm).transpose() << std::endl;
         std::cout << std::endl;
         std::cout << "Ylm    = " << ylm.transpose() << std::endl;
         std::cout << "A *Zlm = " << (A*zlm).transpose() << std::endl;
         std::cout << std::endl;

         std::cout << "Matrix Y:" << std::endl;
         std::cout << Y << std::endl;

         std::cout << "Matrix A:" << std::endl;
         std::cout << A << std::endl;
#endif
      }
   }

   if(nb_fails == 0) {
      std::cout << "Test passed!" << std::endl;
   }
   std::cout << std::endl;
   return nb_fails;
}

/* Compute the integral of the spherical function defined by the SH
 * coefficients `clm` over the spherical triangle using MC.
 */
float MonteCarloSH(const Eigen::VectorXf& clm, const Triangle& triangle) {

   static std::mt19937 gen(0);
   static std::uniform_real_distribution<float> dist(0.0,1.0);

   const int order = sqrt(clm.size());

   // Number of MC samples
   const int M = 10000000;
   float val = 0.0f;
   for(int k=0; k<M; ++k) {

#ifdef USE_TRIANGLE_SAMPLING
      float pdf;
      const Vector d = SampleSphericalTriangle(triangle, pdf);
#else // UNIFORM SAMPLING
      const float pdf = 1.0f / (4.0f*M_PI);
      const Vector d  = Sample();
#endif

      const auto ylm = SHEvalFast<Vector>(d, order-1);

      if(HitTriangle(triangle, d)) {
         val += ylm.dot(clm) / pdf;
      }

   }
   return val / M;
}

/* Check if the integration of the spherical function with SH coeffs `clm`
 * over the spherical triangle `triangle` is the same if done using ZH
 * expansion + Arvo's integral or MC method.
 */
int CheckSHIntegration(const Eigen::VectorXf& clm,
                       const Triangle& tri,
                       float Epsilon = 1.0E-3) {

   std::cout << "Testing the analytical integration with:" << std::endl;
   std::cout << " + Triangle ABC" << std::endl;
   std::cout << "   + A = : " << tri[0].A << std::endl;
   std::cout << "   + B = : " << tri[1].A << std::endl;
   std::cout << "   + C = : " << tri[2].A << std::endl;
   std::cout << " + SH expansion of the integrand" << std::endl;
   std::cout << "   + clm = [" << clm.transpose() << "]" << std::endl;

   const int order = sqrt(clm.size())-1;
   int nb_fails = 0;

   // Monte-Carlo solution
   const auto mcI = MonteCarloSH(clm, tri);

   // Analytical solution
   const auto basis = SamplingBlueNoise<Vector>(2*order+1);
   const auto shI = computeSHIntegral<Triangle, Vector, SH>(clm, basis, tri);

   if(!closeTo(shI, mcI, Epsilon)) {
      ++nb_fails;

      std::cout << "Error: Monte-Carlo = " << mcI
                << " ≠ Analytical = " << shI << std::endl;
   } else {
      std::cout << "Test passed!" << std::endl;
   }

   std::cout << std::endl;
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
   std::vector<Vector> basis;
   std::vector<Vector> queries;
   int order = 2;
   int nbQueryVectors = 20;

   queries = SamplingBlueNoise<Vector>(nbQueryVectors);
   queries.push_back(glm::normalize(glm::vec3(0,0,1)));
   queries.push_back(glm::normalize(glm::vec3(0,1,0)));
   queries.push_back(glm::normalize(glm::vec3(1,0,0)));
   queries.push_back(glm::normalize(glm::vec3(1,1,1)));

   // Using the canonical frame
   basis.clear();
   basis.push_back(glm::normalize(glm::vec3(0,0,1)));
   basis.push_back(glm::normalize(glm::vec3(0,1,0)));
   basis.push_back(glm::normalize(glm::vec3(1,0,0)));
   nb_fails += CheckZHDecomposition(basis, queries);

   // Using random directions
   basis.clear();
   basis = SamplingBlueNoise<Vector>(2*order+1);
   nb_fails += CheckZHDecomposition(basis, queries);

   // Using random directions and order 5
   order = 5;
   basis.clear();
   basis = SamplingBlueNoise<Vector>(2*order+1);
   nb_fails += CheckZHDecomposition(basis, queries);

   // Using random directions and order 10
   order = 10;
   basis.clear();
   basis = SamplingBlueNoise<Vector>(2*order+1);
   nb_fails += CheckZHDecomposition(basis, queries);

   // Using random directions and order 18. This is the maximum order
   // that FastSH can produce.
   order = 18;
   basis.clear();
   basis = SamplingBlueNoise<Vector>(2*order+1);
   nb_fails += CheckZHDecomposition(basis, queries);

   /* SH Integration using the analytical form VS MonteCarlo */
   order = 2;
   int sh_size = (order+1)*(order+1);
   Eigen::VectorXf clm = Eigen::VectorXf::Zero(sh_size);
   clm[0] = 1.0f;
   auto A = glm::vec3(0.0, 0.0, 1.0);
   auto B = glm::vec3(0.0, 0.5, 1.0);
   auto C = glm::vec3(0.5, 0.0, 1.0);
   auto tri = Triangle(glm::normalize(A), glm::normalize(B), glm::normalize(C));
   nb_fails += CheckSHIntegration(clm, tri);

   // Using a random integrand
   clm = Eigen::VectorXf::Random(sh_size).cwiseAbs();
   nb_fails += CheckSHIntegration(clm, tri);

   // Increasing the order to 5
   order = 5;
   sh_size = (order+1)*(order+1);
   clm = Eigen::VectorXf::Random(sh_size).cwiseAbs();
   nb_fails += CheckSHIntegration(clm, tri);

   if(nb_fails == 0) {
      return EXIT_SUCCESS;
   } else {
      return EXIT_FAILURE;
   }
}
