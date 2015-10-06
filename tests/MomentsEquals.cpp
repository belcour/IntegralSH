// Include Moment + test suite
#include "AxialMoments.hpp"
#include "Tests.hpp"

// Include GLM
#include <glm/glm.hpp>

/* Check if all computed moments are equals between the two configurations.
 */
int CheckEquals(const glm::vec3& wA, const Triangle& triA,
                const glm::vec3& wB, const Triangle& triB,
                int nMin, int nMax, float Epsilon = 1.0E-3f) {

   // Track the number of failed tests
   int nb_fails = 0;

   auto momentsA = AxialMoment<Triangle, Vector>(triA, wA, nMax);
   auto momentsB = AxialMoment<Triangle, Vector>(triB, wB, nMax);

   // Test the difference between analytical code and MC
   for(int n=nMin; n<=nMax; ++n) {
      const auto diff = std::abs(momentsA[n] - momentsB[n]);
      if(diff > Epsilon*std::abs(momentsA[n])) {
         ++nb_fails;
         std::cout << "Error for n=" << n << " : "
                   << momentsA[n] << " != " << momentsB[n] << std::endl;
      }
   }

   if(nb_fails == 0) {
      std::cout << "Test passed!" << std::endl;
   }
   return nb_fails;
}


int main(int argc, char** argv) {

   glm::vec3 A, B, C;
   Triangle  triA, triB, triC, triD;
   glm::vec3 wA, wB, wC, wD;

   // Track the number of failed tests
   int nMin = 0, nMax = 10;
   int nb_fails = 0;

   // Configuration A
   A = glm::vec3(0.0, 0.0, 1.0);
   B = glm::vec3(0.1, 0.0, 1.0);
   C = glm::vec3(0.0, 0.1, 1.0);
   triA = Triangle(A, B, C);
   wA   = glm::normalize(glm::vec3(0, 0, 1));

   // Configuration B
   A = glm::vec3( 0.0, 0.0, 1.0);
   B = glm::vec3( 0.0, 0.1, 1.0);
   C = glm::vec3(-0.1, 0.0, 1.0);
   triB = Triangle(A, B, C);
   wB   = glm::normalize(glm::vec3(0, 0, 1));

   // Configuration C
   A = glm::vec3( 0.0, 0.0, 1.0);
   B = glm::vec3(-0.1, 0.0, 1.0);
   C = glm::vec3( 0.0,-0.1, 1.0);
   triC = Triangle(A, B, C);
   wC   = glm::normalize(glm::vec3(0, 0, 1));

   // Configuration C
   A = glm::vec3(0.0, 0.0, 1.0);
   B = glm::vec3(0.0,-0.1, 1.0);
   C = glm::vec3(0.1, 0.0, 1.0);
   triD = Triangle(A, B, C);
   wD   = glm::normalize(glm::vec3(0, 0, 1));

   /* Check for the case where A == B */
   std::cout << "# Check for ABC == ABC'" << std::endl;

   nb_fails += CheckEquals(wA, triA, wA, triA, nMin, nMax);
   std::cout << std::endl;


   /* Keep w = (0,0,1) and rotate the spherical triangle
      by 90, 180, and 270 degress. */
   std::cout << "# Check for w = z and rotate the spherical triangle "
             << "around z" << std::endl;

   // Check for the case where A and B are symmetric
   nb_fails += CheckEquals(wA, triA, wB, triB, nMin, nMax);

   // Check for the case where A and C are symmetric
   nb_fails += CheckEquals(wA, triA, wC, triC, nMin, nMax);

   // Check for the case where A and B are symmetric
   nb_fails += CheckEquals(wA, triA, wD, triD, nMin, nMax);

   // Check for the case where B and C are symmetric
   nb_fails += CheckEquals(wB, triB, wC, triC, nMin, nMax);

   // Check for the case where B and C are symmetric
   nb_fails += CheckEquals(wB, triB, wD, triD, nMin, nMax);

   // Check for the case where A and C are symmetric
   nb_fails += CheckEquals(wC, triC, wD, triD, nMin, nMax);
   std::cout << std::endl;


   /* Make w the first axis of the triangle projected on
      the ground plane. */
   std::cout << "# Check for w = AB and rotate the spherical triangle "
             << "around z" << std::endl;

   wA = glm::vec3( 1, 0, 0);
   wB = glm::vec3( 0, 1, 0);
   wC = glm::vec3(-1, 0, 0);
   wD = glm::vec3( 0,-1, 0);

   // Check for the case where A and B are symmetric
   nb_fails += CheckEquals(wA, triA, wB, triB, nMin, nMax);

   // Check for the case where A and C are symmetric
   nb_fails += CheckEquals(wA, triA, wC, triC, nMin, nMax);

   // Check for the case where A and B are symmetric
   nb_fails += CheckEquals(wA, triA, wD, triD, nMin, nMax);

   // Check for the case where B and C are symmetric
   nb_fails += CheckEquals(wB, triB, wC, triC, nMin, nMax);

   // Check for the case where B and C are symmetric
   nb_fails += CheckEquals(wB, triB, wD, triD, nMin, nMax);

   // Check for the case where A and C are symmetric
   nb_fails += CheckEquals(wC, triC, wD, triD, nMin, nMax);
   std::cout << std::endl;

#ifdef CHECK_ORIENTATION
   // Configuration B is similar to A but with swap vertices
   A = triA[0].A;
   B = triA[2].A;
   C = triA[1].A;
   triB = Triangle(A, B, C);
   wA   = glm::normalize(glm::vec3(0, 0, 1));

   nb_fails += CheckEquals(wA, triA, wA, triB, nMin, nMax);
#endif

   if(nb_fails == 0)
      return EXIT_SUCCESS;
   else
      return EXIT_FAILURE;
}