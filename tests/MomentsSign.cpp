// Include Moment + test suite
#include "AxialMoments.hpp"
#include "Tests.hpp"

// Include GLM
#include <glm/glm.hpp>

/* Check if all computed moments are positive.
 * This happens when the triangle covers a region with <w,v> >= 0 for all v in
 * the triangle.
 */
int CheckPositive(const glm::vec3& w, const Triangle& tri, 
                  int nMin, int nMax) {
   
   std::cout << "Triangle set using:" << std::endl;
   std::cout << " + A = : " << tri[0].A << std::endl;
   std::cout << " + B = : " << tri[1].A << std::endl;
   std::cout << " + C = : " << tri[2].A << std::endl;
   std::cout << std::endl;

   std::cout << "Moment with respect to axis w = " << w << std::endl;
   std::cout << std::endl;

   // Track the number of failed tests
   int nb_fails = 0;

   auto moments = AxialMoment<Triangle, Vector>(tri, w, nMax);

   // Test the difference between analytical code and MC
   for(int n=nMin; n<=nMax; ++n) {
      if(moments[n] < 0.0f) {
         ++nb_fails;
         std::cout << "Error for n=" << n << " : " 
                   << moments[n] << " < 0" << std::endl;
      } else if(std::isnan(moments[n])) {
          ++nb_fails;
          std::cout << "Error for n=" << n << " : "
                    << moments[n] << " is NaN" << std::endl;
      }
   }

   if(nb_fails == 0) {
      std::cout << "Test passed!" << std::endl;
   }
   std::cout << std::endl;
   return nb_fails;
}

/* Check if computed moments alternatve between positive and negative.
 * This happens when the triangle covers a region with <w,v> <= 0 for all v in
 * the triangle.
 */
int CheckAlternate(const glm::vec3& w, const Triangle& tri, 
                   int nMin, int nMax) {
   
   std::cout << "Triangle set using:" << std::endl;
   std::cout << " + A = : " << tri[0].A << std::endl;
   std::cout << " + B = : " << tri[1].A << std::endl;
   std::cout << " + C = : " << tri[2].A << std::endl;
   std::cout << std::endl;

   std::cout << "Moment with respect to axis w = " << w << std::endl;
   std::cout << std::endl;

   // Track the number of failed tests
   int nb_fails = 0;

   auto moments = AxialMoment<Triangle, Vector>(tri, w, nMax);

   // Test the difference between analytical code and MC
   for(int n=nMin; n<=nMax; ++n) {
      if(moments[n] < 0.0f && n % 2 == 0) {
         ++nb_fails;
         std::cout << "Error for n=" << n << " : " 
                   << moments[n] << " < 0" << std::endl;
      } else if(moments[n] > 0.0f && n % 2 == 1) {
         ++nb_fails;
         std::cout << "Error for n=" << n << " : " 
                   << moments[n] << " > 0" << std::endl;
      } else if(std::isnan(moments[n])) {
          ++nb_fails;
          std::cout << "Error for n=" << n << " : "
                    << moments[n] << " is NaN" << std::endl;
      }
   }

   if(nb_fails == 0) {
      std::cout << "Test passed!" << std::endl;
   }
   std::cout << std::endl;
   return nb_fails;
}

int main(int argc, char** argv) {

   const float Eps = 1.0E-5;

   glm::vec3 A, B, C;
   A = glm::vec3(Eps, Eps, 1.0);
   B = glm::vec3(Eps, 0.5, 1.0);
   C = glm::vec3(0.5, Eps, 1.0);
   Triangle  tri(glm::normalize(A), glm::normalize(B), glm::normalize(C));
   glm::vec3 w;
   
   // Track the number of failed tests
   int nMin = 0, nMax = 10;
   int nb_fails = 0;
 
   // Change the moments' axis
   w = glm::normalize(glm::vec3(0, 0, 1));
   nb_fails += CheckPositive(w, tri, nMin, nMax);
   
   w = glm::normalize(glm::vec3(0, 0, -1));
   nb_fails += CheckAlternate(w, tri, nMin, nMax);
    
   w = glm::normalize(glm::vec3(1, 0, 0));
   nb_fails += CheckPositive(w, tri, nMin, nMax);

   w = glm::normalize(glm::vec3(-1, 0, 0));
   nb_fails += CheckAlternate(w, tri, nMin, nMax);

   if(nb_fails == 0)
      return EXIT_SUCCESS;
   else
      return EXIT_FAILURE;
}
