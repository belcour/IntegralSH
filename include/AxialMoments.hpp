#pragma once

// Eigen includes
#include <Eigen/Core>
#include <iostream>

/* _Cosine Sum Integral_
 */
inline Eigen::VectorXf CosSumIntegral(float x, float y, float c, int n) {

	const float siny  = sin(y);
	const float sinx  = sin(x);
	const float cosy  = cos(y);
	const float cosx  = cos(x);
	const float cosy2 = cosy*cosy;
	const float cosx2 = cosx*cosx;

   static const Eigen::Vector2i i1 = {1, 1};
   static const Eigen::Vector2i i2 = {2, 2};
   Eigen::Vector2i i = {0, 1};
   Eigen::Vector2f F = {y-x, siny-sinx};
   Eigen::Vector2f S = {0.0f, 0.0f};

   Eigen::VectorXf R = Eigen::VectorXf::Zero(n+2);

   Eigen::Vector2f pow_c    = {1.0, c};
   Eigen::Vector2f pow_cosx = {cosy, cosy2};
   Eigen::Vector2f pow_cosy = {cosx, cosx2};

   while(i[1] <= n) {
      S += pow_c.cwiseProduct(F);
      R.segment(i[1], 2) = S;

      auto T = pow_cosy*siny - pow_cosx*sinx;
      F = (T + (i+i1).cast<float>().cwiseProduct(F)).cwiseQuotient((i+i2).cast<float>());

      // Update temp variable
      i        += i2;
      pow_c    *= c*c;
      pow_cosx *= cosx2;
      pow_cosy *= cosy2;
   }

   return R;
}

/* Sign function template
 */
template <typename T>
inline int sign(T val) {
    return (T(0) <= val) - (val < T(0));
}

/* Clamp function template to restrict a given function to be in between
 * boundaries.
 */
template <typename T>
inline T clamp(T val, T a, T b) {
    return std::max<T>(a, std::min<T>(b, val));
}


/* _Line Integral_
 */
template<class Vector>
inline Eigen::VectorXf LineIntegral(const Vector& A, const Vector& B,
                                    const Vector& w, int n) {
   auto wDotA = Vector::Dot(A, w);
   auto wDotB = Vector::Dot(B, w);
   if(n < 0 || (wDotA == 0.0f && wDotB == 0.0f)) {
      return Eigen::VectorXf::Zero(n+2);
   }

   // Need check: expanding the (I-ssT)B expression from Arvo's LineIntegral
   auto s = Vector::Normalize(A);
   auto t = Vector::Normalize(B - Vector::Dot(s, B)*s);

   auto a = Vector::Dot(w, s);
   auto b = Vector::Dot(w, t);
   auto c = sqrt(a*a + b*b);

   auto l = acos(Vector::Dot(A, B) / (Vector::Dot(A,A)*Vector::Dot(B,B)));
   auto p = sign(b) * acos(clamp<float>(a / c, -1.0f, 1.0f));

   return CosSumIntegral(-p, l-p, c, n);
}

/* _Boundary Integral_
 *
 * Compute the integral along P egdes of the up to order 'n' axial moment
 * around w. By using 'n' = 'w' you can compute the single axis moment. Double
 * axis moment with second axis begin the normal must use 'v' == 'n' ('n' being
 * the normal).
 */
template<class Polygon, class Vector>
inline Eigen::VectorXf BoundaryIntegral(const Polygon& P, const Vector& w,
                                        const Vector& v, int n) {
   // Init to zero
   Eigen::VectorXf b = Eigen::VectorXf::Zero(n+2);

   for(auto edge : P) {
      // Compute the edge normal
      auto normal = Vector::Normalize(Vector::Cross(edge.A, edge.B));
      // Add the egde integral to the total integral
      b += Vector::Dot(normal, v) * LineIntegral<Vector>(edge.A, edge.B, w, n);
   }

   return b;
}

/* _Solid Angle_
 *
 * Compute the solid angle sustained by a `Polygon P`.
 * TODO: Handle the case where P is more that a Triangle.
 */
template<class Polygon, class Vector>
inline float SolidAngle(const Polygon& P) {
   if(P.size() == 3) {

      const Vector& A = P[0].A;
      const Vector& B = P[1].A;
      const Vector& C = P[2].A;

      const Vector ab = Vector::Normalize(Vector::Cross(A, B));
      const Vector ac = Vector::Normalize(Vector::Cross(A, C));
      const Vector ba = Vector::Normalize(Vector::Cross(B, A));
      const Vector bc = Vector::Normalize(Vector::Cross(B, C));
      const Vector cb = Vector::Normalize(Vector::Cross(C, B));
      return acos(Vector::Dot(ba, ac)) +
             acos(Vector::Dot(cb, ab)) +
             acos(Vector::Dot(ac, bc)) - M_PI;

   } else {
      assert(false);
      return 0.0f;
   }
}

/* _Axial Moments_
 *
 * input:
 *   + Polygon P: A set of egdes that can be enumerated using iterators.
                  Each edge must enable to access two Vector A and B.
 *   + Vector  w: A 3D vector with elements accessible as x,y,z this
                  vector defines the axis on which to compute moments.
 *   + int     n: The maximum moment order to be computed.
 *
 * output:
 *   + VectorX r: A vector containing all moments up to order 'n'
 */
template<class Polygon, class Vector>
inline Eigen::VectorXf AxialMoment(const Polygon& P, const Vector& w, int n) {

   if(n % 2 == 0)
      n = n + 1;

   // Arvo's boundary integral for single vector moment.
   Eigen::VectorXf a = - BoundaryIntegral<Polygon, Vector>(P, w, w, n-1);

   // Generate the 'b' vector which equals to the Polygon solid angle for 
   // even moments and zero for odd moments.
   const int n2 = (n+1)/2;
   auto b = Eigen::Map<Eigen::VectorXf, 0, Eigen::InnerStride<2>>(a.data(), n2); 
   b += Eigen::VectorXf::Constant(n2, SolidAngle<Polygon, Vector>(P));

   // 'c' is the vector of linear elements, storing 'i+1' for index 'i'
   auto c = Eigen::VectorXf::LinSpaced(n+1, 1, n+1);

   return a.cwiseQuotient(c);
}
