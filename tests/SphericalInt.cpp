// STL includes
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <utility>

// Local includes
//#define USE_SPARSE_EIGEN
#include "Tests.hpp"
#include "SH.hpp"
#include "SphericalHarmonics.hpp"
#include "SphericalIntegration.hpp"
#include "DirectionsSampling.hpp"

// GLM include
#include <glm/glm.hpp>

struct CosPowFunctor {
   // Constructor
   int _exp;
   inline CosPowFunctor(int exp=1) : _exp(exp) {}

   // Operator function
   inline float operator()(const Vector& w) const {
      return (_exp+1)*pow(glm::clamp(w.z, 0.0f, 1.0f), _exp) / (2*M_PI);
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
   const Vector n = glm::vec3(0,0,1);
   const Vector w = glm::normalize(glm::vec3(1,1,1));
   const float dotWN = Vector::Dot(w, n);

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
   auto ShBasis = SH::FastBasis(w, lmax);
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
   auto ylm = SH::FastBasis(w, order);
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
   ylm = SH::FastBasis(w, order);
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
   ylm = SH::FastBasis(w, order);
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
   ylm = SH::FastBasis(w, order);
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

   Eigen::VectorXf ylm(SH::Terms(order));

   for(auto& w : queries) {
      // Evaluate the SH function
      SH::FastBasis(w, order, ylm);

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

/* Check matrix mutiplication order
 */
int CheckMatrixOrder(const Eigen::VectorXf& clm,
                     const Triangle& tri) {

   std::cout << "Testing the analytical order" << std::endl;

   const int order = sqrt(clm.size())-1;
   int nb_fails = 0;

   const auto basis = SamplingBlueNoise<Vector>(2*order+1);

   // Get the Zonal weights matrix and the Zlm -> Ylm conversion matrix
   // and compute the product of the two: `Prod = A x Zw`.
   const auto ZW = ZonalWeights<Vector>(basis);
   const auto Y  = ZonalExpansion<SH, Vector>(basis);
   const auto A  = computeInverse(Y);

   const auto Prod = A*ZW;

   // Analytical evaluation of the integral of power of cosines for
   // the different basis elements up to the order defined by the
   // number of elements in the basis
   const Eigen::VectorXf moments = AxialMoments<Triangle, Vector>(tri, basis);

   const float order1 = clm.dot(Prod * moments);
   const float order2 = moments.dot(clm.transpose() * Prod);

   if(!closeTo(order1, order2)) {
      std::cerr << "Error: the evaluation of the integral is order dependant!"
                << std::endl
                << " + clm x (AxZW x mnt) ≠ (clm x AxZW) x mnt" << std::endl
                << " +   " << order1 << " ≠ " << order2 << std::endl
                << std::endl;
      return 1;
   } else {
      return 0;
   }
}

/* Compute the integral of the spherical function defined by the SH
 * coefficients `clm` over the spherical triangle using MC.
 */
std::pair<float,float> MonteCarloSH(const Eigen::VectorXf& clm,
                                    const Triangle& triangle) {

   static std::mt19937 gen(0);
   static std::uniform_real_distribution<float> dist(0.0,1.0);

   const int order = sqrt(clm.size());
   Eigen::VectorXf ylm(clm.size());

   // Number of MC samples
   const int M = 10000000;
   float mean = 0.0f;
   float var  = 0.0f;
   for(int k=0; k<M; ++k) {

#ifdef USE_TRIANGLE_SAMPLING
      float pdf;
      const Vector d = SampleSphericalTriangle(triangle, pdf);
#else // UNIFORM SAMPLING
      const float pdf = 1.0f / (4.0f*M_PI);
      const Vector d  = Sample();
#endif

      SH::FastBasis(d, order, ylm);

      if(HitTriangle(triangle, d)) {
         const auto val = ylm.dot(clm) / pdf;
         mean += val;
         var  += val*val;
      }

   }

   mean /= M;
   var   = var / (M-1) - mean*mean;
   return std::pair<float,float>(mean, 5.0f*sqrt(var/M));
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

   if(!closeTo(shI, mcI)) {
      ++nb_fails;

      std::cout << "Error: Monte-Carlo = " << mcI.first
                << "(±" << mcI.second << ")"
                << " ≠ Analytical = " << shI << std::endl;
   } else {
      std::cout << "Test passed!" << std::endl;
   }

   std::cout << std::endl;
   return nb_fails;
}

template<class SH, class Vector>
Eigen::VectorXf PowerCosineCoeffs(const Vector& w, int n, int order) {

	int n2 = n*n;
	int n3 = n2*n;
	int n4 = n3*n;
	int n5 = n4*n;

	float pi = M_PI;
   auto zhCoeffs = Eigen::VectorXf(19);
	zhCoeffs[0] =            sqrt(pi)  / (1+n);
	zhCoeffs[1] =            sqrt(3*pi) / (2+n);
	zhCoeffs[2] = n          * sqrt(5*pi) / (3+4*n+n2);
	zhCoeffs[3] = (n-1)         * sqrt(7*pi) / (8+6*n+n2);
	zhCoeffs[4] = 3*(n2-2*n)       * sqrt(pi)  / (15+23*n+9*n2+n3);
	zhCoeffs[5] = (3-4*n+n2)       * sqrt(11*pi) / (48+44*n+12*n2+n3);
	zhCoeffs[6] = (n3-6*n2+8*n)       * sqrt(13*pi) / (105+176*n+86*n2+16*n3+n4);
	zhCoeffs[7] = (-15 + 23*n - 9*n2 +n3)    * sqrt(15*pi) / (384 + 400*n + 140*n2 + 20*n3 + n4);
	zhCoeffs[8] = (-48*n + 44*n2 - 12*n3 + n4)   * sqrt(17*pi) / (945 + 1689*n + 950*n2 + 230*n3 + 25*n4 + n5);
	zhCoeffs[9] = (105 - 176*n + 86*n2 - 16*n3 + n4) * sqrt(19*pi) / (3840 + 4384*n + 1800*n2 + 340*n3 + 30*n4 + n5);
	zhCoeffs[10] = ((-8 + n)*(-6 + n)*(-4 + n)*(-2 + n)*n*sqrt(21*pi))            /((1 + n)*(3 + n)*(5 + n)*(7 + n)*(9 + n)*(11 + n));
	zhCoeffs[11] = ((-9 + n)*(-7 + n)*(-5 + n)*(-3 + n)*(-1 + n)*sqrt(23*pi))          /((2 + n)*(4 + n)*(6 + n)*(8 + n)*(10 + n)*(12 + n));
	zhCoeffs[12] = (5*(-10 + n)*(-8 + n)*(-6 + n)*(-4 + n)*(-2 + n)*n*sqrt(pi))          /((1 + n)*(3 + n)*(5 + n)*(7 + n)*(9 + n)*(11 + n)*(13 + n));

	zhCoeffs[13] = 27.009229429366766/(2+ n) - 810.276882881003/(4+ n)
					+ 6887.353504488525/(6 + n) - 24925.660301958473/(8 + n)
					+ 43619.90552842733/(10 + n) - 36482.102805593764/(12 + n)
					+ 11692.981668459539/(14 + n);
	zhCoeffs[14] = -1.99941/(1.+ n) + 209.938/(3. + n)
					- 3568.94/(5. + n) + 22603.3/(7. + n)
					- 67809.9/(9. + n) + 103975./(11. + n)
					- 78769.1/(13. + n) + 23371./(15. + n);
	zhCoeffs[15] = -31.008/(2. + n) + 1229.99/(4. + n)
					- 14021.8/(6. + n) + 70109.2/(8. + n)
					- 179168./(10. + n) + 244320./(12. + n)
					- 169145./(14. + n) + 46716.1/(16. + n);
	zhCoeffs[16] = 1.99954/(1. + n) - 271.938/(3. + n)
					+ 6027.95/(5. + n) - 50634.8/(7. + n)
					+ 207964./(9. + n) - 462143./(11. + n)
					+ 567176./(13. + n) - 361496./(15. + n)
					+ 93386.6/(17. + n);
	zhCoeffs[17] = 35.0071/(2. + n) - 1773.69/(4. + n)
					+ 26073.3/(6. + n) - 171339./(8. + n)
					+ 594927./(10. + n) - 1.16822e6/(12. + n)
					+ 1.30301e6/(14. + n) - 769399./(16. + n)
					+ 186692./(18. + n);
	zhCoeffs[18] = - 1.99964/(1. + n) + 341.938/(3. + n)
					- 9574.25/(5. + n) + 102764./(7. + n)
					- 550520./(9. + n) + 1.65156e6/(11. + n)
					- 2.90274e6/(13. + n) + 2.96654e6/(15. + n)
					- 1.6316e6/(17. + n) + 373241./(19. + n);
   zhCoeffs *= (n+1)/(2*M_PI);
   //zhCoeffs.segment(n+1, 18-n) = Eigen::VectorXf::Zero(18-n);

   const int vsize = (order+1)*(order+1);
   Eigen::VectorXf shCoeffs = SH::FastBasis(w, order);

   Eigen::VectorXf res = Eigen::VectorXf::Zero(vsize);
   for(int l=0; l<=order; ++l) {
      const float factor = sqrt(4*M_PI / (2*l+1)) * zhCoeffs[l];
      for(int m=0; m<2*l+1; ++m) {
         const int i = l*l + m;
         res[i] = factor * shCoeffs[i];
      }
   }
   return res;
}

/* Evaluate the SH decomposition of a phong lobe using Monte-Carlo integration.
 */
template<class SH, class Vector>
Eigen::VectorXf PowerCosineCoeffsMC(const Vector& w, int n, int order) {
   Eigen::VectorXf shCoeffs((order+1)*(order+1));

   int N = 100000;
   for(int i=0; i<N; ++i) {
      const Vector dir = Sample();
      const float cs = glm::clamp(Vector::Dot(w, dir), 0.0f, 1.0f);
      const float fw = (n+1)*pow(cs, n) / (2.0*M_PI);
      shCoeffs += fw * SH::FastBasis(dir, order);
   }
   shCoeffs *= 4.0*M_PI / float(N);

   return shCoeffs;
}

/* Compute the integral of the spherical function defined by the Functor `f`
 * over the spherical triangle using MC.
 */
template<class Functor>
std::pair<float,float> MonteCarlo(const Functor& f,
                                  const Triangle& triangle) {

   static std::mt19937 gen(0);
   static std::uniform_real_distribution<float> dist(0.0,1.0);

   // Number of MC samples
   const int M = 10000000;
   float mean = 0.0f;
   float var  = 0.0f;
   for(int k=0; k<M; ++k) {

#ifdef USE_TRIANGLE_SAMPLING
      float pdf;
      const Vector d = SampleSphericalTriangle(triangle, pdf);
#else // UNIFORM SAMPLING
      const float pdf = 1.0f / (4.0f*M_PI);
      const Vector d  = Sample();
#endif

      if(HitTriangle(triangle, d)) {
         const auto val = f(d) / pdf;
         mean += val;
         var  += val*val;
      }

   }

   mean /= M;
   var   = var / (M-1) - mean*mean;
   return std::pair<float,float>(mean, 5.0f*sqrt(var/M));
}

/* Check if the integration of the spherical function with SH coeffs `clm` over
 * the spherical triangle `triangle` is the same if done using ZH expansion +
 * Arvo's integral or MC method.
 */
template<class Functor>
int CheckSHIntegration(const Eigen::VectorXf& clm,
                       const Functor& f,
                       const Triangle& tri,
                       float Epsilon = 1.0E-3) {

   std::cout << "Testing the analytical integration with:" << std::endl;
   std::cout << " + Triangle ABC" << std::endl;
   std::cout << "   + A = : " << tri[0].A << std::endl;
   std::cout << "   + B = : " << tri[1].A << std::endl;
   std::cout << "   + C = : " << tri[2].A << std::endl;

   const int order = sqrt(clm.size())-1;
   int nb_fails = 0;

   // Monte-Carlo solution
   const auto mcFnI = MonteCarlo<CosPowFunctor>(f, tri);

   // Analytical solution
   const auto basis = SamplingBlueNoise<Vector>(2*order+1);
   const auto shI = computeSHIntegral<Triangle, Vector, SH>(clm, basis, tri);

   if(!closeTo(shI, mcFnI)) {
      ++nb_fails;

      std::cout << "Error: Monte-Carlo (Fn) = " << mcFnI.first
                << "(±" << mcFnI.second << ")"
                << " ≠ Analytical = " << shI << std::endl;
   } else {
      std::cout << "Works! Monte-Carlo (Fn) = " << mcFnI.first
                << "(±" << mcFnI.second << ")"
                << " = Analytical = " << shI << std::endl;
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

#ifdef SKIP
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
#endif

   /* SH Integration using the analytical form VS MonteCarlo */
   order = 2;
   int sh_size = (order+1)*(order+1);
   Eigen::VectorXf clm = Eigen::VectorXf::Zero(sh_size);
   clm[0] = 1.0f;
   auto A = glm::vec3(0.0, 0.0, 1.0);
   auto B = glm::vec3(0.0, 0.5, 1.0);
   auto C = glm::vec3(0.5, 0.0, 1.0);
   auto tri = Triangle(glm::normalize(A), glm::normalize(B), glm::normalize(C));
#ifdef SKIP
   nb_fails += CheckMatrixOrder(clm, tri);
   nb_fails += CheckSHIntegration(clm, tri);

   // Using a random integrand
   clm = Eigen::VectorXf::Random(sh_size).cwiseAbs();
   nb_fails += CheckMatrixOrder(clm, tri);
   nb_fails += CheckSHIntegration(clm, tri);

   // Increasing the order to 5
   order = 5;
   sh_size = (order+1)*(order+1);
   clm = Eigen::VectorXf::Random(sh_size).cwiseAbs();
   nb_fails += CheckMatrixOrder(clm, tri);
   nb_fails += CheckSHIntegration(clm, tri);

   // Changing the triangle
   A = glm::vec3( 0.0, 0.5, 0.5);
   B = glm::vec3(-0.5, 0.5, 0.0);
   C = glm::vec3( 0.5, 0.5, 0.0);
   tri = Triangle(glm::normalize(A), glm::normalize(B), glm::normalize(C));
   nb_fails += CheckMatrixOrder(clm, tri);
   nb_fails += CheckSHIntegration(clm, tri);

   /* SH Integration using the analytical form VS MonteCarlo */
   order = 5;
   const CosPowFunctor f;
   clm = DiffuseCoeffsSH<glm::vec3>(glm::vec3(0,0,1), order);
   A = glm::vec3(0.0, 0.0, 1.0);
   B = glm::vec3(0.0, 0.5, 1.0);
   C = glm::vec3(0.5, 0.0, 1.0);
   tri = Triangle(glm::normalize(A), glm::normalize(B), glm::normalize(C));
   std::cout << "Analytical integration compared to clamped cosine" << std::endl;
   nb_fails += CheckSHIntegration<CosPowFunctor>(clm, f, tri);
#endif


   /* SH Integration using the analytical form VS MonteCarlo */
   int power = 1;
   order = 15;
   CosPowFunctor phong = CosPowFunctor(power);
   basis = SamplingFibonacci<Vector>((order+1)*(order+1));
   clm   = ProjectToSH<CosPowFunctor, Vector, SH>(phong, basis);
   A = glm::vec3(0.0, 0.0, 1.0);
   B = glm::vec3(0.0, 0.5, 1.0);
   C = glm::vec3(0.5, 0.0, 1.0);
   tri = Triangle(glm::normalize(A), glm::normalize(B), glm::normalize(C));
   std::cout << "Analytical integration (using 'ProjectToSH') compared to power cosine "
             << "(" << power << ")" << std::endl;
   nb_fails += CheckSHIntegration<CosPowFunctor>(clm, phong, tri);

   // Exponent 1 Phong lobe using the analytical forms for the ZH coeffs
   phong = CosPowFunctor(power);
   clm   = PowerCosineCoeffsMC<SH, Vector>(Vector(0,0,1), power, order);
   auto clmD = PowerCosineCoeffs<SH, Vector>(Vector(0,0,1), power, order);
//   std::cout << "MC evaluation: " << clm.transpose() << std::endl;
//   std::cout << "An evaluation: " << clmD.transpose() << std::endl;

   std::cout << "Analytical integration (using MC estimation of SH) compared to power cosine "
             << "(" << power << ")" << std::endl;
   nb_fails += CheckSHIntegration<CosPowFunctor>(clm, phong, tri);

   // Exponent 8 Phong lobe
   power = 8;
   phong = CosPowFunctor(power);
   clm   = ProjectToSH<CosPowFunctor, Vector, SH>(phong, basis);
   std::cout << "Analytical integration (using 'ProjectToSH') compared to power cosine "
             << "(" << power << ")" << std::endl;
   nb_fails += CheckSHIntegration<CosPowFunctor>(clm, phong, tri);

   // Exponent 10 Phong lobe
   power = 10;
   phong = CosPowFunctor(power);
   clm   = ProjectToSH<CosPowFunctor, Vector, SH>(phong, basis);
   std::cout << "Analytical integration (using 'ProjectToSH') compared to power cosine "
             << "(" << power << ")" << std::endl;
   nb_fails += CheckSHIntegration<CosPowFunctor>(clm, phong, tri);

   if(nb_fails == 0) {
      return EXIT_SUCCESS;
   } else {
      return EXIT_FAILURE;
   }
}
