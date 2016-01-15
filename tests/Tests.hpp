#pragma once

// Include GLM
#include <glm/glm.hpp>

// Include STL
#include <vector>
#include <random>
#include <utility>
#include <iostream>

// Eigen CORE
#include <Eigen/Core>

// Local include
#include "SH.hpp"

/* _Is `a` close to `b` ?_
 *
 * This template function check if a-b is smaller than a fraction of the max
 * between a and b. By default, it checks if it is smaller than 1 percent of
 * the maximum value.
 */
template<typename T>
inline bool closeTo(const T& a, const T&b, const T& Percentage = T(0.01)) {
   if(a == T(0.0) || b == T(0.0)) {
      return std::abs<T>(a-b) < Percentage;
   } else {
      const T c = std::max<T>(std::max<T>(a, b), Percentage);
      return (std::abs<T>(a-b) < Percentage * c);
   }
}

/* _Is `a` inside the confidence interval `[m-s, m+s]?
 *
 * `b` is supposed to be a confidence interval in the form of µ±ε where µ is
 * `b.first` and ε is `b.second`.
 */
template<typename T>
inline bool closeTo(const T& a, const std::pair<T,T>& b) {
   assert(b.second >= T(0));
   return (a < b.first + b.second) && (a > b.first - b.second);
}

struct Vector : public glm::vec3 {

   Vector() : glm::vec3() {}
   Vector(float x, float y, float z) : glm::vec3(x, y, z) {}
   Vector(const glm::vec3& w) : glm::vec3(w) {}

   static inline float Dot(const glm::vec3& a, const glm::vec3& b) {
      return glm::dot<float>(a, b);
   }

   static inline glm::vec3 Cross(const glm::vec3& a, const glm::vec3& b) {
      return glm::cross(a, b);
   }

   static inline glm::vec3 Normalize(const glm::vec3& a) {
      return glm::normalize(a);
   }

   static inline float Length(const glm::vec3& a) {
      return glm::length(a);
   }
};

std::ostream& operator<< (std::ostream& out, const glm::vec3& a) {
   out << "[" << a.x << ", " << a.y << ", " << a.z << "]";
   return out;
}


struct Edge {
   Edge(const Vector& a, const Vector& b) : A(a), B(b) {}
   Vector A, B;
};

/* 'The Triangle' structure represent a spherical triangle of three coordinates
 * A, B and C by storing its Edges in a vcetor.
 *
 * TODO: Extend to the polygon structure
 */
struct Triangle : public std::vector<Edge> {
   Triangle() {
   }
   Triangle(const Vector& A, const Vector& B, const Vector& C) {
     this->push_back(Edge(A, B));
     this->push_back(Edge(B, C));
     this->push_back(Edge(C, A));
   }
};

struct Quad: public std::vector<Edge> {
   Quad() {
   }
   Quad(const Vector& A, const Vector& B,
        const Vector& C, const Vector& D) {
     this->push_back(Edge(A, B));
     this->push_back(Edge(B, C));
     this->push_back(Edge(C, D));
     this->push_back(Edge(D, A));
   }
};

struct Polygon : public std::vector<Edge> {

   // Constructor
   Polygon() {
   }
   Polygon(const Vector& A, const Vector& B, const Vector& C) {
      this->push_back(Edge(A, B));
      this->push_back(Edge(B, C));
      this->push_back(Edge(C, A));
   }
};

std::ostream& operator<<(std::ostream& out, const Polygon& polygon) {
   for(auto& edge : polygon) {
      out << " + " << edge.A << std::endl;
   }
   return out;
}

struct PolygonConstructor : public std::vector<Vector> {

   // Constructor
   PolygonConstructor() {
   }
   PolygonConstructor(const Vector& A, const Vector& B, const Vector& C) {
      this->push_back(A);
      this->push_back(B);
      this->push_back(C);
   }

   Polygon ProjectToHemisphere(const Vector& p) const {
      Polygon P;
#ifdef REMOVE
      std::cout << "A = " << P[0].A << std::endl;
      std::cout << "B = " << P[1].A << std::endl;
      std::cout << "C = " << P[2].A << std::endl;
#endif
      for(unsigned int k=0; k<this->size(); ++k) {
         const Vector& A = this->at(k);
         const Vector& B = (k == this->size()-1) ? this->at(0) : this->at(k+1);

         P.push_back(Edge(Vector::Normalize(A-p), Vector::Normalize(B-p)));
      }
      return P;
   }

   /* Clamp the Polygon with respect to the Shading normal.
    */
   Polygon ProjectToHemisphere(const Vector& p, const Vector& n) const {
      Polygon P;
#ifdef REMOVE
      std::cout << "A = " << P[0].A << std::endl;
      std::cout << "B = " << P[1].A << std::endl;
      std::cout << "C = " << P[2].A << std::endl;
#endif
      // Constant
      const unsigned int size = this->size();

      // Starting vector of the Edge. This vector can be clamped if necessary
      // to account for the shading horizon.
      unsigned int start = 0;
      Vector A, M;
      float dotAn;
      bool condition = true;
      do {
         // Get the current element
         A = Vector::Normalize(this->at(start) - p);

         // Initial condition: the vertex needs to be above the horizon
         dotAn = Vector::Dot(A, n);
         condition = dotAn < 0.0;
         ++start;

      } while(condition && start<size);
      --start;

      // Return the empty polygon in case every vertex is below the horizon
      if(start == size) { return P; }

      for(unsigned int k=1; k<=size; ++k) {
         const unsigned int next = (start + k) % size;
         const Vector B = Vector::Normalize(this->at(next) - p);

         const float dotBn = Vector::Dot(B, n);

         // First case: the beginning of the Edge was below the hemisphere.
         // Then we must create the intermediate point N and create an egde
         // using M and N and another with N and B..
         if(dotAn < 0 && dotBn >= 0) {
            const float alpha = dotAn / (dotAn - dotBn);
            const Vector N  = Vector::Normalize(A + alpha*(B-A));

            if(Vector::Dot(M, N) < 1) {
               P.push_back(Edge(M, N));
            }
            if(alpha > 0) {
               P.push_back(Edge(N, B));
            }

         } else if(dotAn >= 0 && dotBn < 0) {
            const float alpha = dotAn / (dotAn - dotBn);
            M = Vector::Normalize(A + alpha*(B-A));
            if(alpha > 0) {
               P.push_back(Edge(A, M));
            }

         // The next point is a valid one (above the horizon). Add the Edge
         // (A,B)
         } else if(dotAn >= 0 && dotBn >= 0) {
            P.push_back(Edge(A, B));
         }

         // Update the loop variables.
         A = B;
         dotAn = dotBn;
      }
      std::cout << std::endl;
      return P;
   }
};

std::mt19937 _test_gen(0);
std::uniform_real_distribution<float> _test_dist(0.0,1.0);

/* 'Sample' generate a random direction on the unit sphere with uniform
 * distribution using _test_gen and _test_dist random number generators..
 */
glm::vec3 Sample() {

   glm::vec3 out;

   // Sample the cosine of the elevation
   const double z = _test_dist(_test_gen);
   out.z = 2.0*z - 1.0;
   const double z2 = out.z*out.z;

   // Sample the azimuth
   const double phi = 2.0*M_PI*_test_dist(_test_gen);
   out.x = sqrt(1.0-z2) * cos(phi);
   out.y = sqrt(1.0-z2) * sin(phi);
   return out;
}

// Sample a spherical triangle using Arvo's stratified method.
// Note: This code is not reliable right now. An error in the computation of the
// solid angle bias the distribution of points. Use the uniform sampling method
// instead.
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

   const float phi = _test_dist(_test_gen)*area - alpha;
   const float t   = cos(phi);
   const float s   = sin(phi);
   const float u   = t - cos(alpha);
   const float v   = s + sin(alpha)*glm::dot(A, B);

   const float q = (v*t - u*s)*cos(alpha) - v / ((v*s + u*t)*sin(alpha));

   glm::vec3 hC = q*A + float(sqrt(1.0f-q*q))*glm::normalize(C-glm::dot(C, A)*A);

   // Select the cos(theta)
   const float z = 1.0f - _test_dist(_test_gen)*(1.0f - glm::dot(hC, B));

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
   auto p   = Vector::Cross(w, e2);
   auto det = Vector::Dot(e1, p);

   //if determinant is near zero, ray lies in plane of triangle otherwise not
   if (det > -Epsilon && det < Epsilon) { return false; }
   auto invDet = 1.0f / det;

   //calculate distance from p1 to ray origin
   auto t = - p1;

   //Calculate u parameter
   auto u = Vector::Dot(t, p) * invDet;

   //Check for ray hit
   if (u < 0 || u > 1) { return false; }

   //Prepare to test v parameter
   auto q = glm::cross(t, e1);

   //Calculate v parameter
   auto v = Vector::Dot(w, q) * invDet;

   //Check for ray hit
   if (v < 0 || u + v > 1) { return false; }

   if ((Vector::Dot(e2, q) * invDet) > Epsilon) {
       //ray does intersect
       return true;
   }

   // No hit at all
   return false;
}

/* Spherical Harmonics wrapper for the code in 'SphericalHarmonics.hpp'
 */
struct SH {

   // Inline for FastBasis
   static inline Eigen::VectorXf FastBasis(const Vector& w, int lmax) {
      const auto size = Terms(lmax);
      Eigen::VectorXf res(size);
      SHEvalFast<Vector>(w, lmax, res);
      return res;
   }
   static inline void FastBasis(const Vector& w, int lmax, Eigen::VectorXf& clm) {
      assert(clm.size() == Terms(lmax));
      SHEvalFast<Vector>(w, lmax, clm);
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
