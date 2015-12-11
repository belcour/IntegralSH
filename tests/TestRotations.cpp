// STL includes
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <utility>
#include <thread>

// Local includes
#include "SH.hpp"
#include "Tests.hpp"
#include "SphericalHarmonics.hpp"
#include "DirectionsSampling.hpp"
#include "SphericalIntegration.hpp"
#include "SHRotation.hpp"

// GLM include
#include <glm/glm.hpp>

int CheckQuaternion() {

   Eigen::Matrix3f rot;
   rot << 1.0, 0.0, 0.0,
          0.0, 0.0,-1.0,
          0.0, 1.0, 0.0;

   Eigen::Quaternionf quat(rot);

   std::cout << rot << std::endl << std::endl;
   std::cout << quat.toRotationMatrix() << std::endl << std::endl;
   std::cout << quat.norm() << std::endl << std::endl;

   return 0;
}

int CheckRotation(int order = 5, int nbTrials = 10) {
   int nb_fails = 0;

   auto axisRot    = Eigen::AngleAxisf(0.25*M_PI, Eigen::Vector3f::UnitX());
   auto quaternion = Eigen::Quaternionf(axisRot);
   if(!closeTo(quaternion.norm(), 1.0f)) {
      std::cerr << "Fail! The Quaternion is not normalized." << std::endl;
      return ++nb_fails;
   }
   auto rotation   = Rotation(order, quaternion);

   Eigen::VectorXf clm  = Eigen::VectorXf::Random(SHTerms(order));
   Eigen::VectorXf rclm = Eigen::VectorXf::Zero(SHTerms(order));
   rotation.Apply(clm, rclm);

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

int CheckRotationExplicitMatrix(int order = 5, int nbTrials = 10) {
   int nb_fails = 0;

   Eigen::Matrix3f rot;
   rot << 1.0, 0.0, 0.0,
          0.0, 0.0,-1.0,
          0.0, 1.0, 0.0;
   auto quaternion = Eigen::Quaternionf(rot);
   if(!closeTo(quaternion.norm(), 1.0f)) {
      std::cerr << "Fail! The Quaternion is not normalized." << std::endl;
      return ++nb_fails;
   }
   auto rotation   = Rotation(order, quaternion);

   Eigen::VectorXf clm  = Eigen::VectorXf::Random(SHTerms(order));
   Eigen::VectorXf rclm = Eigen::VectorXf::Zero(SHTerms(order));
   rotation.Apply(clm, rclm);

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

   CheckQuaternion();

   // Check the rotation code with multiple threads
   std::vector<std::thread> threads;
   for(int k=0; k<10; ++k) {
      threads.push_back(std::thread(CheckRotation, k, 20));
   }
   for(auto& thread : threads) {
      thread.join();
   }

   // Check another method to init the quaternion
   CheckRotationExplicitMatrix(5, 20);

   if(nb_fails > 0) {
      return EXIT_FAILURE;
   } else {
      return EXIT_SUCCESS;
   }
}
