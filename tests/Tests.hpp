#pragma once

// Include GLM
#include <glm/glm.hpp>

// Include STL
#include <vector>

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

std::ostream& operator<< (std::ostream& out, const glm::vec3& a) {
   out << "[" << a.x << ", " << a.y << ", " << a.z << "]";
   return out;
}
