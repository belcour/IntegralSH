#pragma once

// Include GLM
#include <glm/glm.hpp>

// Include STL
#include <vector>
#include <random>
#include <utility>

// Eigen CORE
// Required for the template version of abs
#include <Eigen/Core>

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

struct Edge {
   Edge(const glm::vec3& a, const glm::vec3& b) : A(a), B(b) {}
   glm::vec3 A, B;
};

struct Triangle : public std::vector<Edge> {
   Triangle() {
   }
   Triangle(const glm::vec3& A, const glm::vec3& B, const glm::vec3& C) {
     this->push_back(Edge(A, B));
     this->push_back(Edge(B, C));
     this->push_back(Edge(C, A));
   }
};

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

std::mt19937 _test_gen(0);
std::uniform_real_distribution<float> _test_dist(0.0,1.0);

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
