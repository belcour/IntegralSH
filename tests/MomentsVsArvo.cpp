#include "AxialMoments.hpp"
#include "ArvoMoments.hpp"
#include "Tests.hpp"

// Include GLM
#include <glm/glm.hpp>

// Include STL
#include <iostream>
#include <vector>

int TestSolidAngle(const Triangle& tri, float Epsilon = 1.0E-5) {

   std::cout << "Triangle set using:" << std::endl;
   std::cout << " + A = : " << tri[0].A << std::endl;
   std::cout << " + B = : " << tri[1].A << std::endl;
   std::cout << " + C = : " << tri[2].A << std::endl;
   std::cout << std::endl;

   // Track the number of failed tests
   int nb_fails = 0;

   auto analytical = SolidAngle<Triangle, Vector>(tri);
   std::cout << "Analytical solid angle : " << analytical << std::endl;

   float tri_array[9] = {tri[0].A.x, tri[0].A.y, tri[0].A.z,
                         tri[1].A.x, tri[1].A.y, tri[1].A.z,
                         tri[2].A.x, tri[2].A.y, tri[2].A.z };

   auto arvo = TriangleSolidAngle(&tri_array[0], &tri_array[3], &tri_array[6]);
   std::cout << "Arvo's solid angle : " << arvo << std::endl;

   if(std::abs(analytical - arvo) > Epsilon*std::abs(analytical) ||
      std::isnan(analytical)) {
      std::cerr << "Error: solid angle differs from Arvo's!" << std::endl;
      std::cerr << "       error is = " << std::abs(analytical - arvo) << std::endl;
      ++nb_fails;
   }

   std::cout << std::endl;

   return nb_fails;
}

int TestSolidAngle(const Quad& quad, float Epsilon = 1.0E-5) {

   std::cout << "Quad set using:" << std::endl;
   std::cout << " + A = : " << quad[0].A << std::endl;
   std::cout << " + B = : " << quad[1].A << std::endl;
   std::cout << " + C = : " << quad[2].A << std::endl;
   std::cout << " + D = : " << quad[3].A << std::endl;
   std::cout << std::endl;

   // Track the number of failed tests
   int nb_fails = 0;

   auto analytical = SolidAngle<Quad, Vector>(quad);
   std::cout << "Analytical solid angle : " << analytical << std::endl;

   float quad_array[9] = {quad[0].A.x, quad[0].A.y, quad[0].A.z,
                          quad[1].A.x, quad[1].A.y, quad[1].A.z,
                          quad[2].A.x, quad[2].A.y, quad[2].A.z };
   float arvo = TriangleSolidAngle(&quad_array[0], &quad_array[3], &quad_array[6]);
   float sec_array[9] = {quad[2].A.x, quad[2].A.y, quad[2].A.z,
                         quad[3].A.x, quad[3].A.y, quad[3].A.z,
                         quad[0].A.x, quad[0].A.y, quad[0].A.z };
   arvo += TriangleSolidAngle(&sec_array[0], &sec_array[3], &sec_array[6]);
   std::cout << "Arvo's solid angle : " << arvo << std::endl;

   if(std::abs(analytical - arvo) > Epsilon*std::abs(analytical) ||
      std::isnan(analytical)) {
      std::cerr << "Error: solid angle differs from Arvo's!" << std::endl;
      std::cerr << "       error is = " << std::abs(analytical - arvo) << std::endl;
      ++nb_fails;
   }

   std::cout << std::endl;

   return nb_fails;
}

int TestMoments(const glm::vec3& w, const Triangle& tri,
                int nMin, int nMax,
                float Epsilon = 1.0E-5) {

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
      auto analytical = moments[n];
      std::cout << "Analytical for n=" << n << " : " << analytical << std::endl;

      float tri_array[9] = {tri[0].A.x, tri[0].A.y, tri[0].A.z,
                            tri[1].A.x, tri[1].A.y, tri[1].A.z,
                            tri[2].A.x, tri[2].A.y, tri[2].A.z };
      float w_array[3] = {w.x, w.y, w.z};

      auto arvo = ArvoAxialMoment(tri_array, w_array, n);
      std::cout << "Arvo's for n=" << n << " : " << arvo << std::endl;

      if(std::abs(analytical - arvo) > Epsilon*std::abs(analytical) ||
         std::isnan(analytical)) {
         std::cerr << "Error: moment " << n << " differs from Arvo's!" << std::endl;
         std::cerr << "       error is = " << std::abs(analytical - arvo) << std::endl;
         ++nb_fails;
      }
   }
   std::cout << std::endl;

   return nb_fails;
}

int main(int argc, char** argv) {

   // Track the number of failed tests
   float Eps = 1.0E-5, Epsilon = 1.0E-2;
   int nMin = 0, nMax = 10;
   int nb_fails = 0;


   // Generate a triangle + lobe direction configuration
   glm::vec3 A, B, C, D, w;
   Triangle tri;


   /* Check the computation of the solid angle */
   A = glm::vec3( 0.5,-0.5, 1.0);
   B = glm::vec3(-0.5,-0.5, 1.0);
   C = glm::vec3( 0.0, 0.5, 1.0);
   tri = Triangle(glm::normalize(A), glm::normalize(B), glm::normalize(C));
   nb_fails += TestSolidAngle(tri, Epsilon);

   // Shfited triangle on the right upper quadrant
   A = glm::vec3(Eps, Eps, 1.0);
   B = glm::vec3(Eps, 0.5, 1.0);
   C = glm::vec3(0.5, Eps, 1.0);
   tri = Triangle(glm::normalize(A), glm::normalize(B), glm::normalize(C));
   nb_fails += TestSolidAngle(tri, Epsilon);

   A = glm::vec3( 0.5,  0.0, 1.0);
   B = glm::vec3( 0.0,  0.5, 1.0);
   C = glm::vec3(-0.5,  0.0, 1.0);
   D = glm::vec3( 0.0, -0.5, 1.0);
   Quad quad = Quad(glm::normalize(A), glm::normalize(B),
                    glm::normalize(C), glm::normalize(D));
   nb_fails += TestSolidAngle(quad, Epsilon);

   /* Moment computation comparison */

   // Change the moments' axis
   A = glm::vec3(Eps, Eps, 1.0);
   B = glm::vec3(Eps, 0.5, 1.0);
   C = glm::vec3(0.5, Eps, 1.0);
   tri = Triangle(glm::normalize(A), glm::normalize(B), glm::normalize(C));
   w = glm::normalize(glm::vec3(0, 0, 1));
   nb_fails += TestMoments(w, tri, nMin, nMax, Epsilon);

   w = glm::normalize(glm::vec3(0, 0, -1));
   nb_fails += TestMoments(w, tri, nMin, nMax, Epsilon);

   w = glm::normalize(glm::vec3(1, 0, 1));
   nb_fails += TestMoments(w, tri, nMin, nMax, Epsilon);

   w = glm::normalize(glm::vec3(1, 0, 0));
   nb_fails += TestMoments(w, tri, nMin, nMax, Epsilon);

   // Change the triangle slightly but change the same axis.
   A = glm::vec3(0.0, 0.0, 1.0);
   B = glm::vec3(0.0, 0.5, 1.0);
   C = glm::vec3(0.5, 0.0, 1.0);
   tri = Triangle(glm::normalize(A), glm::normalize(B), glm::normalize(C));
   nb_fails += TestMoments(w, tri, nMin, nMax, Epsilon);

   // Change the triangle
   A = glm::vec3(0.00, 0.00, 1.0);
   B = glm::vec3(0.00, 0.1, 1.0);
   C = glm::vec3(0.01, 0.00, 1.0);
   tri = Triangle(glm::normalize(A), glm::normalize(B), glm::normalize(C));
   w = glm::normalize(glm::vec3(0.05,0.05,1));
   nb_fails += TestMoments(w, tri, nMin, nMax, Epsilon);

   // Check the case where Nmax is odd
   nMax = 11;
   A = glm::vec3(Eps, Eps, 1.0);
   B = glm::vec3(Eps, 0.5, 1.0);
   C = glm::vec3(0.5, Eps, 1.0);
   tri = Triangle(glm::normalize(A), glm::normalize(B), glm::normalize(C));

   // Change the moments' axis
   w = glm::normalize(glm::vec3(0, 0, 1));
   nb_fails += TestMoments(w, tri, nMin, nMax, Epsilon);

   w = glm::normalize(glm::vec3(0, 0, -1));
   nb_fails += TestMoments(w, tri, nMin, nMax, Epsilon);

   // Integrate a full quadrant
   nMax = 10;
   A = glm::vec3(0.0, 0.0, 1.0);
   B = glm::vec3(0.0, 1.0, 0.0);
   C = glm::vec3(1.0, 0.0, 0.0);
   tri = Triangle(glm::normalize(A), glm::normalize(B), glm::normalize(C));

   w = glm::normalize(glm::vec3(0, 0, 1));
   nb_fails += TestMoments(w, tri, nMin, nMax, Epsilon);

   w = glm::normalize(glm::vec3(1, 1, 1));
   nb_fails += TestMoments(w, tri, nMin, nMax, Epsilon);

   if(nb_fails == 0)
      return EXIT_SUCCESS;
   else
      return EXIT_FAILURE;
}
