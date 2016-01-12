#include "AxialMoments.hpp"
#include "DirectionsSampling.hpp"

//#define USE_TRIANGLE_SAMPLING
#include "Tests.hpp"

#include <glm/glm.hpp>

#include <iostream>
#include <vector>

std::mt19937 gen(0);
std::uniform_real_distribution<float> dist(0.0,1.0);

typedef std::pair<Eigen::VectorXf,Eigen::VectorXf> VectorXfPair;

VectorXfPair MonteCarloMoments(const Triangle& triangle,
                               const Vector& w, int n) {

   // Number of MC samples
   const int M = 10000000;
   Eigen::VectorXf mean = Eigen::VectorXf::Zero(n+1);
   Eigen::VectorXf var  = Eigen::VectorXf::Zero(n+1);
   for(int k=0; k<M; ++k) {
#ifdef USE_TRIANGLE_SAMPLING
      float pdf;
      glm::vec3 d = SampleSphericalTriangle(triangle, pdf);
#else // UNIFORM SAMPLING
      float pdf = 1.0f / (4.0f*M_PI);
      glm::vec3 d = Sample();
#endif

      if(HitTriangle(triangle, d)) {
         for(int p=0; p<=n; ++p) {
            const auto val = pow(Vector::Dot(d, w), p) / pdf;
            mean[p] += val;
            var[p]  += val*val;
         }
      }
   }

   mean  = mean / float(M);
   var   = (var / float(M-1) - mean.cwiseProduct(mean)) / float(M);
   return VectorXfPair(mean, 5.0f*var.cwiseSqrt());
}

template<class Polygon>
int TestMoments(const glm::vec3& w, const Polygon& polygon,
                int nMin, int nMax,
                float Epsilon = 1.0E-5) {

   std::cout << "Polygon set using:" << std::endl;
   for(unsigned int k=0; k<polygon.size(); ++k)
      std::cout << " + " << char('A'+k) << " = : " << polygon[k].A << std::endl;
   std::cout << std::endl;

   std::cout << "Moment with respect to axis w = " << w << std::endl;
   std::cout << std::endl;

   // Track the number of failed tests
   int nb_fails = 0;

   Eigen::VectorXf moments;
   if(polygon.size() == 3) {
      Triangle tri(polygon[0].A, polygon[1].A, polygon[2].A);
      moments = AxialMoment<Triangle, Vector>(tri, w, nMax);
   } else if(polygon.size() == 4) {
      Quad quad(polygon[0].A, polygon[1].A, polygon[2].A, polygon[3].A);
      moments = AxialMoment<Quad, Vector>(quad, w, nMax);

      Triangle tri(polygon[0].A, polygon[1].A, polygon[2].A);
      Eigen::VectorXf temp = AxialMoment<Triangle, Vector>(tri, w, nMax);
      tri = Triangle(polygon[2].A, polygon[3].A, polygon[0].A);
      temp += AxialMoment<Triangle, Vector>(tri, w, nMax);
      if(! moments.isApprox(temp)) {
         ++nb_fails;
         std::cerr << "Error: the moments computation on quads is not the sum of triangles" << std::endl;
         std::cout << " + Quad => " << moments.transpose() << std::endl;
         std::cout << " + Tri  => " << temp.transpose() << std::endl;
      }
   }

   // Compute the moments using Monte-Carlo
   VectorXfPair mc;
   if(polygon.size() == 3) {
      Triangle tr = Triangle(polygon[0].A, polygon[1].A, polygon[2].A);
      mc = MonteCarloMoments(tr, w, nMax);
   } else if(polygon.size() == 4) {
      Triangle tr = Triangle(polygon[0].A, polygon[1].A, polygon[2].A);
      mc = MonteCarloMoments(tr, w, nMax);
      //std::cout << mc.first << std::endl << std::endl;
      tr = Triangle(polygon[1].A, polygon[2].A, polygon[3].A);
      auto temp = MonteCarloMoments(tr, w, nMax);
      //std::cout << temp.first << std::endl << std::endl;
      mc.first  += temp.first;
      mc.second += temp.second;
   }

   // Test the difference between analytical code and MC
   for(int n=nMin; n<=nMax; ++n) {
      auto analytical = moments[n];
      auto mcI = std::pair<float,float>(mc.first[n], mc.second[n]);
      std::cout << "Analytical for n=" << n << " : " << analytical << std::endl;

      std::cout << "MonteCarlo for n=" << n << " : " << mcI.first
                                       << " ± " << mcI.second << std::endl;

      if(!closeTo(analytical, mcI) || std::isnan(analytical)) {
         std::cerr << "Error: moment " << n << " differs from MC!" << std::endl;
         std::cerr << "       error is = " << std::abs(analytical - mcI.first) << std::endl;
         ++nb_fails;
      }
   }
   std::cout << std::endl;

   return nb_fails;
}

std::pair<float,float> MonteCarloSolidAngle(const Triangle& triangle) {

   // Number of MC samples
   const int M = 10000000;
   float mean  = 0.0f;
   float var   = 0.0f;
   float fact  = 4.0f*M_PI;
   for(int k=0; k<M; ++k) {

      // Sample a direction on the sphere
      glm::vec3 d = Sample();

      if(HitTriangle(triangle, d)) {
         mean += fact;
         var  += fact*fact;
      }

   }
   mean = mean / float(M);
   var  = var / float(M-1) - mean*mean;
   return std::pair<float,float>(mean, 5.0f*sqrt(var/M));
}

template<class Polygon>
int TestSolidAngle(const Polygon& polygon, float Epsilon = 1.0E-5) {

   std::cout << "Polygon set using:" << std::endl;
   for(unsigned int k=0; k<polygon.size(); ++k)
      std::cout << " + " << char('A'+k) << " = : " << polygon[k].A << std::endl;
   std::cout << std::endl;

   // Track the number of failed tests
   int nb_fails = 0;

   auto analytical = SolidAngle<Polygon, Vector>(polygon);
   std::cout << "Analytical solid angle : " << analytical << std::endl;

   std::pair<float, float> mc;
   if(polygon.size() == 3) {
      Triangle tr = Triangle(polygon[0].A, polygon[1].A, polygon[2].A);
      mc  = MonteCarloSolidAngle(tr);
   } else if(polygon.size() == 4) {
      Triangle tr = Triangle(polygon[0].A, polygon[1].A, polygon[2].A);
      mc  = MonteCarloSolidAngle(tr);
      tr  = Triangle(polygon[2].A, polygon[3].A, polygon[0].A);
      auto temp = MonteCarloSolidAngle(tr);
      mc.first  += temp.first;
      mc.second += temp.second;
   }
   std::cout << "MC solid angle : " << mc.first
                                    << " ± " << mc.second << std::endl;

   if(!closeTo(analytical, mc) || std::isnan(analytical)) {
      std::cerr << "Error: solid angle differs from MC!" << std::endl;
      std::cerr << "       error is = " << std::abs(analytical - mc.first) << std::endl;
      ++nb_fails;
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
   Triangle tri; Quad quad;


   // Check on Quads with the unit upper direction and randomly choosen
   // directions.
   A = glm::vec3( 0.5,-0.5, 1.0);
   B = glm::vec3(-0.5,-0.5, 1.0);
   C = glm::vec3(-0.5, 0.5, 1.0);
   D = glm::vec3( 0.5, 0.5, 1.0);
   quad = Quad(glm::normalize(A), glm::normalize(B), glm::normalize(C), glm::normalize(D));
   w = glm::normalize(glm::vec3(0, 0, 1));
   nb_fails += TestMoments<Quad>(w, quad, nMin, nMax, Epsilon);
   w = glm::normalize(glm::vec3(1, 0, 0));
   nb_fails += TestMoments<Quad>(w, quad, nMin, nMax, Epsilon);

   /* Check the solid angle */

   // Check on Triangle
   A = glm::vec3( 0.5,-0.5, 1.0);
   B = glm::vec3(-0.5,-0.5, 1.0);
   C = glm::vec3( 0.0, 0.5, 1.0);
   tri = Triangle(glm::normalize(A), glm::normalize(B), glm::normalize(C));
   nb_fails += TestSolidAngle<Triangle>(tri, Epsilon);

   // Check on Quad
   C = glm::vec3(-0.5, 0.5, 1.0);
   D = glm::vec3( 0.5, 0.5, 1.0);
   quad = Quad(glm::normalize(A), glm::normalize(B), glm::normalize(C), glm::normalize(D));
   nb_fails += TestSolidAngle<Quad>(quad, Epsilon);

   /* Check the moments */

   // Shfited triangle on the right upper quadrant
   A = glm::vec3(Eps, Eps, 1.0);
   B = glm::vec3(Eps, 0.5, 1.0);
   C = glm::vec3(0.5, Eps, 1.0);
   tri = Triangle(glm::normalize(A), glm::normalize(B), glm::normalize(C));

   // Change the moments' axis
   w = glm::normalize(glm::vec3(0, 0, 1));
   nb_fails += TestMoments<Triangle>(w, tri, nMin, nMax, Epsilon);

   w = glm::normalize(glm::vec3(0, 0, -1));
   nb_fails += TestMoments<Triangle>(w, tri, nMin, nMax, Epsilon);

   w = glm::normalize(glm::vec3(1, 0, 1));
   nb_fails += TestMoments<Triangle>(w, tri, nMin, nMax, Epsilon);

   w = glm::normalize(glm::vec3(1, 0, 0));
   nb_fails += TestMoments<Triangle>(w, tri, nMin, nMax, Epsilon);

   // Change the triangle slightly but change the same axis.
   A = glm::vec3(0.0, 0.0, 1.0);
   B = glm::vec3(0.0, 0.5, 1.0);
   C = glm::vec3(0.5, 0.0, 1.0);
   tri = Triangle(glm::normalize(A), glm::normalize(B), glm::normalize(C));
   nb_fails += TestMoments<Triangle>(w, tri, nMin, nMax, Epsilon);

   // Change the triangle
   A = glm::vec3(0.00, 0.00, 1.0);
   B = glm::vec3(0.00, 0.1, 1.0);
   C = glm::vec3(0.01, 0.00, 1.0);
   tri = Triangle(glm::normalize(A), glm::normalize(B), glm::normalize(C));
   w = glm::normalize(glm::vec3(0.05,0.05,1));
   nb_fails += TestMoments<Triangle>(w, tri, nMin, nMax, Epsilon);

   // Check the case where Nmax is odd
   nMax = 11;
   A = glm::vec3(Eps, Eps, 1.0);
   B = glm::vec3(Eps, 0.5, 1.0);
   C = glm::vec3(0.5, Eps, 1.0);
   tri = Triangle(glm::normalize(A), glm::normalize(B), glm::normalize(C));

   // Change the moments' axis
   w = glm::normalize(glm::vec3(0, 0, 1));
   nb_fails += TestMoments<Triangle>(w, tri, nMin, nMax, Epsilon);

   w = glm::normalize(glm::vec3(0, 0, -1));
   nb_fails += TestMoments<Triangle>(w, tri, nMin, nMax, Epsilon);

   // Integrate a full quadrant
   nMax = 10;
   A = glm::vec3(0.0, 0.0, 1.0);
   B = glm::vec3(0.0, 1.0, 0.0);
   C = glm::vec3(1.0, 0.0, 0.0);
   tri = Triangle(glm::normalize(A), glm::normalize(B), glm::normalize(C));

   w = glm::normalize(glm::vec3(0, 0, 1));
   nb_fails += TestMoments<Triangle>(w, tri, nMin, nMax, Epsilon);

   w = glm::normalize(glm::vec3(1, 1, 1));
   nb_fails += TestMoments<Triangle>(w, tri, nMin, nMax, Epsilon);

   // Add random direction with a small triangle with centroid being
   // the z-vector.
   A = glm::vec3( 0.5,-0.5, 1.0);
   B = glm::vec3(-0.5,-0.5, 1.0);
   C = glm::vec3( 0.0, 0.5, 1.0);
   tri = Triangle(glm::normalize(A), glm::normalize(B), glm::normalize(C));

   for(int nb_rand=0; nb_rand<10; ++nb_rand) {
      w = glm::normalize(glm::vec3(2.0f*(dist(gen) - 0.5f), 2.0f*(dist(gen) - 0.5f), 2.0f*(dist(gen) - 0.5f)));
      nb_fails += TestMoments<Triangle>(w, tri, nMin, nMax, Epsilon);
   }

   // Check on Quads with the unit upper direction and randomly choosen
   // directions.
   A = glm::vec3( 0.5,-0.5, 1.0);
   B = glm::vec3(-0.5,-0.5, 1.0);
   C = glm::vec3(-0.5, 0.5, 1.0);
   D = glm::vec3( 0.5, 0.5, 1.0);

   // Test for normal direction
   quad = Quad(glm::normalize(A), glm::normalize(B), glm::normalize(C), glm::normalize(D));
   w = glm::normalize(glm::vec3(0, 0, 1));

   // Test for a grazing direction. This should give zero moments for
   // odd orders.
   nb_fails += TestMoments<Quad>(w, quad, nMin, nMax, Epsilon);
   w = glm::normalize(glm::vec3(1, 0, 0));

   // Random direction testing.
   nb_fails += TestMoments<Quad>(w, quad, nMin, nMax, Epsilon);
   for(int nb_rand=0; nb_rand<10; ++nb_rand) {
      w = glm::normalize(glm::vec3(2.0f*(dist(gen) - 0.5f), 2.0f*(dist(gen) - 0.5f), 2.0f*(dist(gen) - 0.5f)));
      nb_fails += TestMoments<Quad>(w, quad, nMin, nMax, Epsilon);
   }

   if(nb_fails == 0)
      return EXIT_SUCCESS;
   else
      return EXIT_FAILURE;
}
