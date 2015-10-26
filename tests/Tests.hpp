#pragma once

// Include GLM
#include <glm/glm.hpp>

// Include STL
#include <vector>

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
  const T c = std::max<T>(std::max<T>(a, b), Percentage);
  return (std::abs<T>(a-b) < Percentage * c);
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
      return glm::dot(a, b);
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
