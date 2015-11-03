#pragma once

#include <vector>
#include <random>
#include <cmath>
#include <limits>
#include <chrono>

/* Compute the minimum dot distance between a set of directions and another
 * direction. The vectors are assumed to be normalized here.
 */
template<class Vector>
float MinDotDistance(const std::vector<Vector>& dirs, const Vector& w) {

   // The set of testing direction is empty.
   if(dirs.size() == 0) {
      return 0.0f;
   }

   // Got a least one direction in `dirs`
   float dist = std::numeric_limits<float>::max();
   for(auto& d : dirs) {
      const float dot = Vector::Dot(d, w);
      dist = std::min(dist, dot);
   }
   return dist;
}

/* _Sampling Blue Noise Directions_
 */
template<class Vector>
std::vector<Vector> SamplingBlueNoise(int nb, int MAX_TRY = 1000) {

   std::mt19937 gen(std::chrono::system_clock::now().time_since_epoch().count());
   std::uniform_real_distribution<float> dist(0.0,1.0);

   std::vector<Vector> res;
   res.reserve(nb);

   float max_dist = 0.5;
   int n=0;
   while(n<nb) {

      // Vector to add
      Vector w;

      int nb_try = 0;
      while(nb_try < MAX_TRY) {

         // Generate a random direction by uniformly sampling the sphere.
         float z = 2.0f * dist(gen) - 1.0f;
         float y = sqrt(1.0f - z*z);
         float p = 2.0f * M_PI * dist(gen);
         w = Vector::Normalize(Vector(y*cos(p), y*sin(p), z));

         // Testing if the distance satisfy blue noise properties.
         float dot_dist = MinDotDistance(res, w);
         if(dot_dist < max_dist) {
            break;
         } else {
            ++nb_try;
         }
      }

      if(nb_try == MAX_TRY) {
         max_dist = sqrt(max_dist);
      } else {
         res.push_back(w);
         ++n;
      }
   }

   return res;
}

/* _Sampling Fibonacci Directions_
 */
template<class Vector>
std::vector<Vector> SamplingFibonacci(int nb) {

   // Golden ratio
   const float gratio = (sqrt(5.0f)+1.0f)/2.0f;

   std::vector<Vector> res;
   res.reserve(nb);

   for(int i=0; i<nb; ++i) {
      const float zi = 1.0f - float(2*i+1) / float(nb);
      const float xi = float(i) / gratio;
      const float pi = 2.0f*M_PI * (xi - std::floor(xi));

      const float sint = sqrt(1.0 - zi*zi);

      res.push_back(Vector(sint*cos(pi), sint*sin(pi), zi));
   }
   return res;
}
