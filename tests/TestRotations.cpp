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
#include "SHRotation.hpp"

// GLM include
#include <glm/glm.hpp>


int CheckRotation(int order = 5, int nbTrials = 10) {
   int nb_fails = 0;

   auto axisRot    = Eigen::AngleAxisf(0.25*M_PI, Eigen::Vector3f::UnitX());
   auto quaternion = Eigen::Quaternionf(axisRot);
   auto rotation   = Rotation::Create(order, quaternion);
   if(quaternion.norm() != 1.0f) {
      std::cerr << "Fail! The Quaternion is not normalized." << std::endl;
   }

   Eigen::VectorXf clm  = Eigen::VectorXf::Random(SHTerms(order));
   Eigen::VectorXf rclm = Eigen::VectorXf::Zero(SHTerms(order));
   rotation->Apply(clm, rclm);

   const auto dirs = SamplingFibonacci<Eigen::Vector3f>(nbTrials);
   for(auto& w : dirs) {
      const Eigen::VectorXf ylm  = SHEvalFast<Eigen::Vector3f>(w, order);
      const Eigen::VectorXf rylm = SHEvalFast<Eigen::Vector3f>(quaternion._transformVector(w), order);

      const float v  = ylm.dot(clm);
      const float rv = rylm.dot(rclm);
      if(! closeTo(v, rv)) {
         std::cerr << "for direction " << w.transpose() << ": "
                   << v << " ≠ " << rv << std::endl;
         nb_fails++;
      }
   }

   return nb_fails;
}


int main(int argc, char** argv) {

   int nb_fails = 0;

   nb_fails += CheckRotation(2);
   nb_fails += CheckRotation(5);
   nb_fails += CheckRotation(8);

   if(nb_fails > 0) {
      return EXIT_FAILURE;
   } else {
      return EXIT_SUCCESS;
   }
}
