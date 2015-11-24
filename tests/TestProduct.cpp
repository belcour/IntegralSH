// STL includes
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <utility>

// Local includes
#include "SH.hpp"
#include "Tests.hpp"
#include "SphericalHarmonics.hpp"
#include "DirectionsSampling.hpp"
#include "SphericalIntegration.hpp"

// GLM include
#include <glm/glm.hpp>

struct SH {

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

struct ProductSH {

   const Eigen::VectorXf _flm, _glm;
   const int _order;

   ProductSH(const Eigen::VectorXf& flm, const Eigen::VectorXf& glm) :
      _flm(flm), _glm(glm), _order(sqrt(glm.size())-1) {
      assert(glm.size() == flm.size());
   }

   float operator()(const Vector& w) const {
      const auto ylm = SH::FastBasis(w, _order);
      return _flm.dot(ylm) * _glm.dot(ylm);
   }
};

/* Compute the decomposition of a shading cosine on the SH basis up to 19
 * coefficients.
 */
Eigen::VectorXf DiffuseSHDecomposition(int order) {
   assert(order < 19);

   const float pisqrt = sqrt(M_PI);
   Eigen::VectorXf zhCoeffs(19);
	zhCoeffs[0]  = pisqrt/2.0;
	zhCoeffs[1]  = sqrt(M_PI/3.0);
	zhCoeffs[2]  = sqrt(5.0*M_PI)/8.0;
	zhCoeffs[3]  = 0.0;
	zhCoeffs[4]  = -pisqrt/16.0;
	zhCoeffs[5]  = 0.0;
	zhCoeffs[6]  = sqrt(13.0*M_PI)/128.0;
	zhCoeffs[7]  = 0.0;
	zhCoeffs[8]  = -sqrt(17.0*M_PI)/256.0;
	zhCoeffs[9]  = 0.0;
	zhCoeffs[10] = 7.0*sqrt(7.0*M_PI/3.0)/1024.0;
	zhCoeffs[11] = 0.0;
	zhCoeffs[12] = -15.0*pisqrt/2048;
	zhCoeffs[13] = 0.0;
	zhCoeffs[14] = 33.0*sqrt(29.0*M_PI)/32768.0;
	zhCoeffs[15] = 0.0;
	zhCoeffs[16] = -143.0*sqrt(11.0*M_PI/3.0)/65536.0;
	zhCoeffs[17] = 0.0;
	zhCoeffs[18] = 143.0*sqrt(37.0*M_PI)/262144.0;

   const int vsize = (order+1)*(order+1);
   Eigen::VectorXf shCoeffs = Eigen::VectorXf::Zero(vsize);
   for(int l=0; l<order; ++l) {
      const int index = (l+1)*l;
      shCoeffs[index] = zhCoeffs[l] / M_PI;
   }

   return shCoeffs;
}

/* Take the product of a random function with a shading cosine and perform the
 * integral of this function over a spherical triangle. Compare this with the
 * Monte-Carlo integral.
 */
int IntegrateProduct() {
   int nb_fails = 0;

   const bool trunc = false;
   const int order  = 9;
   const int morder = trunc ? order : 2*order-1;
   const int nsize = SH::Terms(order);
   const int msize = trunc ? nsize : SH::Terms(morder);
   const Eigen::VectorXf cosYlm = DiffuseSHDecomposition(order);
   Eigen::VectorXf fYlm = cosYlm;//Eigen::VectorXf::Random(SH::Terms(order)).cwiseAbs();
   //fYlm(0) = 1.0;

   // Precompute the matrix precomputed triple tensor product to evaluate the
   // product of (f · cos)(w) using SH.
   std::cout << "# Precomputing the TripleTensorProduct" << std::endl;
   const Eigen::MatrixXf fMat    = TripleTensorProduct<SH, Vector>(fYlm, trunc);
   const Eigen::VectorXf cosFYlm = fMat.block(0, 0, msize, nsize) * cosYlm;

   const int nDirections = 100;
   std::cout << "# Testing against " << nDirections
             << " [quasi]-random directions" << std::endl;
   const std::vector<Vector> directions = SamplingFibonacci<Vector>(nDirections);
   for(auto& w : directions) {
      const auto ylm = SH::FastBasis(w, morder);
      const auto rlm = ylm.segment(0, nsize);

      // Compute the product of f and cos using the individual values
      const float cosVal  = cosYlm.dot(rlm);
      const float fVal    = fYlm.dot(rlm);
      const float prodVal = cosVal*fVal;

      // Compute the product of f and cos using the precomputed product
      const float altVal  = cosFYlm.dot(ylm);

      if(! closeTo(prodVal, altVal)) {
         ++nb_fails;
         std::cout << "# for direction " << w << ": "
                   << prodVal << " ≠ " << altVal << std::endl;
      }
   }

   if(nb_fails == 0) {
      std::cout << "# Test passed!" << std::endl;
   } else {
      std::cout << "# Test failed!" << std::endl;
   }
   std::cout << std::endl;
   return nb_fails;
}

/* Take the product of a random function with a shading cosine and perform the
 * integral of this function over a spherical triangle. Compare this with the
 * Monte-Carlo integral.
 */
int IntegrateProducts() {
   int nb_fails = 0;

   const bool trunc = false;
   const int order  = 9;
   const int morder = trunc ? order : 2*order-1;
   const int nsize = SH::Terms(order);
   const int msize = trunc ? nsize : SH::Terms(morder);
   const Eigen::VectorXf cosYlm = DiffuseSHDecomposition(order);
   std::vector<Eigen::VectorXf> fYlms;
   fYlms.push_back(0.1 * cosYlm);//Eigen::VectorXf::Random(SH::Terms(order)).cwiseAbs();
   fYlms.push_back(1.0 * cosYlm);//Eigen::VectorXf::Random(SH::Terms(order)).cwiseAbs();
   fYlms.push_back(2.0 * cosYlm);//Eigen::VectorXf::Random(SH::Terms(order)).cwiseAbs();

   std::vector<Eigen::VectorXf> cosFYlms;

   // Precompute the matrix precomputed triple tensor product to evaluate the
   // product of (f · cos)(w) using SH.
   std::cout << "# Precomputing the TripleTensorProduct" << std::endl;
   const auto fMats = TripleTensorProduct<SH, Vector>(fYlms, trunc);
   for(const auto& fMat : fMats) {
      cosFYlms.push_back(fMat.block(0, 0, msize, nsize) * cosYlm);
   }

   const int nDirections = 100;
   std::cout << "# Testing against " << nDirections
             << " [quasi]-random directions" << std::endl;
   const std::vector<Vector> directions = SamplingFibonacci<Vector>(nDirections);
   for(auto& w : directions) {
      const auto ylm = SH::FastBasis(w, morder);
      const auto rlm = ylm.segment(0, nsize);

      // Compute the product of f and cos using the individual values
      const float cosVal  = cosYlm.dot(rlm);
      std::vector<float> prodVals;
      for(const auto& fYlm : fYlms) {
         const float fVal    = fYlm.dot(rlm);
         prodVals.push_back(cosVal*fVal);
      }

      // Compute the product of f and cos using the precomputed product
      std::vector<float> altVals;
      for(const auto& cosFYlm : cosFYlms) {
         altVals.push_back(cosFYlm.dot(ylm));
      }

      assert(altVals.size() == prodVals.size());
      for(auto i=0; i<altVals.size(); ++i) {
         if(! closeTo(prodVals[i], altVals[i])) {
            ++nb_fails;
            std::cout << "# for direction " << w << ": "
                      << prodVals[i] << " ≠ " << altVals[i] << std::endl;
         }
      }
   }

   if(nb_fails == 0) {
      std::cout << "# Test passed!" << std::endl;
   } else {
      std::cout << "# Test failed!" << std::endl;
   }
   std::cout << std::endl;
   return nb_fails;
}

int main(int argc, char** argv) {

   int nb_fails = 0;

   // Load an example
   nb_fails += IntegrateProduct();
   nb_fails += IntegrateProducts();

   if(nb_fails > 0) {
      return EXIT_FAILURE;
   } else {
      return EXIT_SUCCESS;
   }
}
