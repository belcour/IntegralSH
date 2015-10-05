#include "AxialMoments.hpp"

#include <glm/glm.hpp>

#include <iostream>
#include <vector>
#include <random>

struct Edge {
   Edge(const glm::vec3& a, const glm::vec3& b) : A(a), B(b) {}
   glm::vec3 A, B;
};

struct Triangle : public std::vector<Edge> {
   Triangle(const glm::vec3& A, const glm::vec3& B, const glm::vec3& C) {
     this->push_back(Edge(A, B));
     this->push_back(Edge(B, C));
     this->push_back(Edge(C, A));
   }
};

struct Vector : public glm::vec3 {

   Vector(const glm::vec3& w) : glm::vec3(w) {}

   static inline float Dot(const glm::vec3& a, const glm::vec3& b) {
      return glm::dot(a, b);
   }

   static inline glm::vec3 Cross(const glm::vec3& a, const glm::vec3& b) {
      return glm::cross(a, b);
   }

   static inline glm::vec3 Normalize(const glm::vec3& a) {
      return glm::normalize(a);
   }
};

std::mt19937 gen(0);
std::uniform_real_distribution<float> dist(0.0,1.0);

glm::vec3 Sample() {

   glm::vec3 out;

   // Sample the cosine of the elevation
   const double z = dist(gen);
   out.z = 2.0*z - 1.0;
   const double z2 = out.z*out.z;

   // Sample the azimuth
   const double phi = 2.0*M_PI*dist(gen);
   out.x = sqrt(1.0-z2) * cos(phi);
   out.y = sqrt(1.0-z2) * sin(phi);
   return out;
}

// Sample a spherical triangle using Arvo's stratified method.
glm::vec3 SampleSphericalTriangle(const Triangle& triangle, float& pdf) {
   const glm::vec3& A = triangle[0].A;
   const glm::vec3& B = triangle[1].A;
   const glm::vec3& C = triangle[2].A;

   const glm::vec3 ab = glm::normalize(glm::cross(A, B));
   const glm::vec3 ac = glm::normalize(glm::cross(A, C));
   const glm::vec3 ba = glm::normalize(glm::cross(B, A));
   const glm::vec3 bc = glm::normalize(glm::cross(B, C));
   const glm::vec3 cb = glm::normalize(glm::cross(C, B));

   const float alpha = acos(glm::dot(ba, ac));
   const float beta  = acos(glm::dot(cb, ab));
   const float gamma = acos(glm::dot(ac, bc));

   const float area  = alpha + beta + gamma - M_PI;
   pdf = 1.0f / area;

   const float phi = dist(gen)*area - alpha;
   const float t   = cos(phi);
   const float s   = sin(phi);
   const float u   = t - cos(alpha);
   const float v   = s + sin(alpha)*glm::dot(A, B);

   const float q = (v*t - u*s)*cos(alpha) - v / ((v*s + u*t)*sin(alpha));

   glm::vec3 hC = q*A + float(sqrt(1.0f-q*q))*glm::normalize(C-glm::dot(C, A)*A);

   // Select the cos(theta)
   const float z = 1.0f - dist(gen)*(1.0f - glm::dot(hC, B));

   return z*B + float(sqrt(1.0f-z*z))*glm::normalize(hC - glm::dot(hC, B)*B);
}

bool HitTriangle(const Triangle& triangle, const Vector& w) {

   const float Epsilon = 1.0E-6;

   auto& p1 = triangle[0].A;
   auto& p2 = triangle[0].B;
   auto& p3 = triangle[1].B;

   //Find vectors for two edges sharing vertex/point p1
   auto e1 = p2 - p1;
   auto e2 = p3 - p1;

   // calculating determinant
   auto p   = glm::cross(w, e2);
   auto det = glm::dot(e1, p);

   //if determinant is near zero, ray lies in plane of triangle otherwise not
   if (det > -Epsilon && det < Epsilon) { return false; }
   auto invDet = 1.0f / det;

   //calculate distance from p1 to ray origin
   auto t = - p1;

   //Calculate u parameter
   auto u = glm::dot(t, p) * invDet;

   //Check for ray hit
   if (u < 0 || u > 1) { return false; }

   //Prepare to test v parameter
   auto q = glm::cross(t, e1);

   //Calculate v parameter
   auto v = glm::dot(w, q) * invDet;

   //Check for ray hit
   if (v < 0 || u + v > 1) { return false; }

   if ((glm::dot(e2, q) * invDet) > Epsilon) {
       //ray does intersect
       return true;
   }

   // No hit at all
   return false;
}

float MonteCarloMoments(const Triangle& triangle, const Vector& w, int n) {

   glm::vec3 A = triangle[0].B - triangle[0].A; // edge 0
   glm::vec3 B = triangle[1].B - triangle[1].A; // edge 1
   glm::vec3 N = glm::normalize(glm::cross(A, B)); // this is the triangle's normal
   //float D = glm::dot(N, triangle[0].A);

   // Number of MC samples
   const int M = 10000;
   float val = 0.0f;
   for(int k=0; k<M; ++k) {
      float pdf;
      glm::vec3 d = SampleSphericalTriangle(triangle, pdf);

      if(HitTriangle(triangle, d)) {
         val += pow(glm::dot(d, w), n) / pdf;
      }

   }
   return val / M;
}

std::ostream& operator<< (std::ostream& out, const glm::vec3& a) {
   out << "[" << a.x << ", " << a.y << ", " << a.z << "]";
   return out;
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

   // Test the difference between analytical code and MC
   for(int n=nMin; n<=nMax; ++n) {
      auto analytical = AxialMoment<Triangle, Vector>(tri, w, n)[0];
      std::cout << "Analytical for n=" << n << " : " << analytical << std::endl;

      auto montecarlo = MonteCarloMoments(tri, w, n);
      std::cout << "MonteCarlo for n=" << n << " : " << montecarlo << std::endl;

      if(std::abs(analytical - montecarlo) > Epsilon) {
         std::cerr << "Error: moment " << n << " differs from MC!" << std::endl;
         ++nb_fails;
      }
   }
   std::cout << std::endl;

   return nb_fails;
}

int main(int argc, char** argv) {

   glm::vec3 A, B, C;
   A = glm::vec3(0.0, 0.0, 1.0);
   B = glm::vec3(0.5, 0.0, 1.0);
   C = glm::vec3(0.0, 0.5, 1.0);

   Triangle  tri(glm::normalize(A), glm::normalize(B), glm::normalize(C));
   glm::vec3 w = glm::normalize(glm::vec3(0,0,1));

   // Track the number of failed tests
   float Epsilon = 1.0E-3;
   int nMin = 0, nMax = 10;
   int nb_fails = 0;
  
   // Change the moments' axis 
   nb_fails += TestMoments(w, tri, nMin, nMax, Epsilon);

   w = glm::normalize(glm::vec3(1, 0, 1));
   nb_fails += TestMoments(w, tri, nMin, nMax, Epsilon);

   if(nb_fails == 0)
      return EXIT_SUCCESS;
   else
      return EXIT_FAILURE;
}
