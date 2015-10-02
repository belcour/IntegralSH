// Eigen includes
#include <Eigen/Core>

/* TODO: Change the output type from Eigen::VectorXf to a template type
 * with possible predefined 'n'.
 */

/* _Cosine Sum Integral_
 */
inline Eigen::VectorXf CosSumIntegral(float x, float y, float c, int n) {

	const float siny  = sin(y);
	const float sinx  = sin(x);
	const float cosy  = cos(y);
	const float cosx  = cos(x);
	const float cosy2 = cosy*cosy;
	const float cosx2 = cosx*cosx;

   const bool is_n_even = n % 2 == 0;
   auto i = (is_n_even) ? 0 : 1;
   auto F = (is_n_even) ? y-x : siny-sinx;

   Eigen::VectorXf S = Eigen::VectorXf::Zero(1);

   auto pow_c    = (is_n_even) ? 1.0  : c;
   auto pow_cosy = (is_n_even) ? cosy : cosy2;
   auto pow_cosx = (is_n_even) ? cosx : cosx2;

   while(i <= n) {
      S[0] = ((i < 2) ? 0.0 : S[0]) + pow_c * F;

      auto T = pow_cosy*siny - pow_cosx*sinx;
      F = (T + (i+1)*F) / (i+2);

      // Update temp variable
      i        += 2;
      pow_c    *= c*c;
      pow_cosx *= cosx2;
      pow_cosy *= cosy2;
   }

   return S;
}

/* Sign function template
 */
template <typename T>
inline int sign(T val) {
    return (T(0) <= val) - (val < T(0));
}


/* _Line Integral_
 */
template<class Vector>
inline Eigen::VectorXf LineIntegral(const Vector& A, const Vector& B,
                                    const Vector& w, int n) {
   auto wDotA = Vector::Dot(A, w);
   auto wDotB = Vector::Dot(B, w);
   if(n < 0 || (wDotA == 0.0f && wDotB == 0.0f)) {
      return Eigen::VectorXf::Zero(1);
   }

   // Need check: expanding the (I-ssT)B expression from Arvo's LineIntegral
   auto s = Vector::Normalize(A);
   auto t = Vector::Normalize(B - Vector::Dot(s, B)*s);

   auto a = Vector::Dot(w, s);
   auto b = Vector::Dot(w, t);
   auto c = sqrt(a*a + b*b);

   auto l = acos(Vector::Dot(A, B) / (Vector::Dot(A,A)*Vector::Dot(B,B)));
   auto p = sign(b) * acos(a / c);

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
   Eigen::VectorXf b = Eigen::VectorXf::Zero(1);

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

      const Vector a = Vector::Normalize(Vector::Cross(P[0].A, P[1].A));
      const Vector b = Vector::Normalize(Vector::Cross(P[1].A, P[2].A));
      const Vector c = Vector::Normalize(Vector::Cross(P[2].A, P[0].A));
      return acos(Vector::Dot(a, b)) +
             acos(Vector::Dot(b, c)) +
             acos(Vector::Dot(c, a)) - M_PI;

   } else {
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
   Eigen::VectorXf a = - BoundaryIntegral<Polygon, Vector>(P, w, w, n-1);
   if(n % 2 == 0) {
      auto b = Eigen::VectorXf(a.size());
      b.fill(SolidAngle<Polygon, Vector>(P));
      return (a + b) / float(n+1);
   } else {
      return a / float(n+1);
   }
}
