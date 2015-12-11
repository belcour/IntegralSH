#pragma once
/* This code is a stripped down version of 'google/spherical-harmonics' on
 * github to focus on spherical harmonics rotations using Eigen. This part
 * of the code is release using the Apache license V2.0 (see [here][license])
 *
 * Modifications:
 *  + I added a new Apply interface to the Rotation class that takes Eigen
 *    vectors as input and outputs to perform efficient Matrix products.
 *    See Rotation::Apply(const Eigen::VectorXf&, Eigen::VectorXf&) const.
 *  + I changed all Eigen type to be 32b float instead of 64b.
 *
 * [license]: https://github.com/google/spherical-harmonics/blob/master/LICENSE
 */

// Include Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

// Include STL
#include <vector>
#include <memory>

template <class T>
using VectorX = Eigen::Matrix<T, Eigen::Dynamic, 1>;

// Get the total number of coefficients for a function represented by
// all spherical harmonic basis of degree <= @order (it is a point of
// confusion that the order of an SH refers to its degree and not the order).
constexpr int GetCoefficientCount(int order) {
  return (order + 1) * (order + 1);
}

// Get the one dimensional index associated with a particular degree @l
// and order @m. This is the index that can be used to access the Coeffs
// returned by SHSolver.
constexpr int GetIndex(int l, int m) {
  return l * (l + 1) + m;
}

// Usage: CHECK(bool, string message);
// Note that it must end a semi-colon, making it look like a
// valid C++ statement (hence the awkward do() while(false)).
#ifndef NDEBUG
# define CHECK(condition, message) \
  do { \
    if (!(condition)) { \
      std::cerr << "Check failed (" #condition ") in " << __FILE__ \
        << ":" << __LINE__ << ", message: " << message << std::endl; \
      std::exit(EXIT_FAILURE); \
    } \
  } while(false)
#else
# define ASSERT(condition, message) do {} while(false)
#endif

// Return true if the first value is within epsilon of the second value.
bool NearByMargin(double actual, double expected) {
  double diff = actual - expected;
  if (diff < 0.0) {
    diff = -diff;
  }
  // 5 bits of error in mantissa (source of '32 *')
  return diff < 32 * std::numeric_limits<double>::epsilon();
}

// ---- The following functions are used to implement SH rotation computations
//      based on the recursive approach described in [1, 4]. The names of the
//      functions correspond with the notation used in [1, 4].

// See http://en.wikipedia.org/wiki/Kronecker_delta
double KroneckerDelta(int i, int j) {
  if (i == j) {
    return 1.0;
  } else {
    return 0.0;
  }
}

// [4] uses an odd convention of referring to the rows and columns using
// centered indices, so the middle row and column are (0, 0) and the upper
// left would have negative coordinates.
//
// This is a convenience function to allow us to access an Eigen::MatrixXf
// in the same manner, assuming r is a (2l+1)x(2l+1) matrix.
double GetCenteredElement(const Eigen::MatrixXf& r, int i, int j) {
  // The shift to go from [-l, l] to [0, 2l] is (rows - 1) / 2 = l,
  // (since the matrix is assumed to be square, rows == cols).
  int offset = (r.rows() - 1) / 2;
  return r(i + offset, j + offset);
}

// P is a helper function defined in [4] that is used by the functions U, V, W.
// This should not be called on its own, as U, V, and W (and their coefficients)
// select the appropriate matrix elements to access (arguments @a and @b).
double P(int i, int a, int b, int l, const std::vector<Eigen::MatrixXf>& r) {
  if (b == l) {
    return GetCenteredElement(r[1], i, 1) *
        GetCenteredElement(r[l - 1], a, l - 1) -
        GetCenteredElement(r[1], i, -1) *
        GetCenteredElement(r[l - 1], a, -l + 1);
  } else if (b == -l) {
    return GetCenteredElement(r[1], i, 1) *
        GetCenteredElement(r[l - 1], a, -l + 1) +
        GetCenteredElement(r[1], i, -1) *
        GetCenteredElement(r[l - 1], a, l - 1);
  } else {
    return GetCenteredElement(r[1], i, 0) * GetCenteredElement(r[l - 1], a, b);
  }
}

// The functions U, V, and W should only be called if the correspondingly
// named coefficient u, v, w from the function ComputeUVWCoeff() is non-zero.
// When the coefficient is 0, these would attempt to access matrix elements that
// are out of bounds. The list of rotations, @r, must have the @l - 1
// previously completed band rotations. These functions are valid for l >= 2.

double U(int m, int n, int l, const std::vector<Eigen::MatrixXf>& r) {
  // Although [1, 4] split U into three cases for m == 0, m < 0, m > 0
  // the actual values are the same for all three cases
  return P(0, m, n, l, r);
}

double V(int m, int n, int l, const std::vector<Eigen::MatrixXf>& r) {
  if (m == 0) {
    return P(1, 1, n, l, r) + P(-1, -1, n, l, r);
  } else if (m > 0) {
    return P(1, m - 1, n, l, r) * sqrt(1 + KroneckerDelta(m, 1)) -
        P(-1, -m + 1, n, l, r) * (1 - KroneckerDelta(m, 1));
  } else {
    // Note there is apparent errata in [1,4,4b] dealing with this particular
    // case. [4b] writes it should be P*(1-d)+P*(1-d)^0.5
    // [1] writes it as P*(1+d)+P*(1-d)^0.5, but going through the math by hand,
    // you must have it as P*(1-d)+P*(1+d)^0.5 to form a 2^.5 term, which
    // parallels the case where m > 0.
    return P(1, m + 1, n, l, r) * (1 - KroneckerDelta(m, -1)) +
        P(-1, -m - 1, n, l, r) * sqrt(1 + KroneckerDelta(m, -1));
  }
}

double W(int m, int n, int l, const std::vector<Eigen::MatrixXf>& r) {
  if (m == 0) {
    // whenever this happens, w is also 0 so W can be anything
    return 0.0;
  } else if (m > 0) {
    return P(1, m + 1, n, l, r) + P(-1, -m - 1, n, l, r);
  } else {
    return P(1, m - 1, n, l, r) - P(-1, -m + 1, n, l, r);
  }
}

// Calculate the coefficients applied to the U, V, and W functions. Because
// their equations share many common terms they are computed simultaneously.
void ComputeUVWCoeff(int m, int n, int l, double* u, double* v, double* w) {
  double d = KroneckerDelta(m, 0);
  double denom = (abs(n) == l ? 2.0 * l * (2.0 * l - 1) : (l + n) * (l - n));

  *u = sqrt((l + m) * (l - m) / denom);
  *v = 0.5 * sqrt((1 + d) * (l + abs(m) - 1.0) * (l + abs(m)) / denom)
      * (1 - 2 * d);
  *w = -0.5 * sqrt((l - abs(m) - 1) * (l - abs(m)) / denom) * (1 - d);
}

// Calculate the (2l+1)x(2l+1) rotation matrix for the band @l.
// This uses the matrices computed for band 1 and band l-1 to compute the
// matrix for band l. @rotations must contain the previously computed l-1
// rotation matrices, and the new matrix for band l will be appended to it.
//
// This implementation comes from p. 5 (6346), Table 1 and 2 in [4] taking
// into account the corrections from [4b].
void ComputeBandRotation(int l, std::vector<Eigen::MatrixXf>* rotations) {
  // The band's rotation matrix has rows and columns equal to the number of
  // coefficients within that band (-l <= m <= l implies 2l + 1 coefficients).
  Eigen::MatrixXf rotation(2 * l + 1, 2 * l + 1);
  for (int m = -l; m <= l; m++) {
    for (int n = -l; n <= l; n++) {
      double u, v, w;
      ComputeUVWCoeff(m, n, l, &u, &v, &w);

      // The functions U, V, W are only safe to call if the coefficients
      // u, v, w are not zero
      if (!NearByMargin(u, 0.0))
          u *= U(m, n, l, *rotations);
      if (!NearByMargin(v, 0.0))
          v *= V(m, n, l, *rotations);
      if (!NearByMargin(w, 0.0))
          w *= W(m, n, l, *rotations);

      rotation(m + l, n + l) = (u + v + w);
    }
  }

  rotations->push_back(rotation);
}

class Rotation {
 public:
/*
  // Create a new Rotation that can applies @rotation to sets of coefficients
  // for the given @order. @order must be at least 0.
  static std::unique_ptr<Rotation> Create(int order,
                                          const Eigen::Quaternionf& rotation);

  // Create a new Rotation that applies the same rotation as @rotation. This
  // can be used to efficiently calculate the matrices for the same 3x3
  // transform when a new order is necessary.
  static std::unique_ptr<Rotation> Create(int order, const Rotation& rotation);
*/
  // Transform the SH basis coefficients in @coeff by this rotation and store
  // them into @result. These may be the same vector. The @result vector will
  // be resized if necessary, but @coeffs must have its size equal to
  // GetCoefficientCount(order()).
  //
  // This rotation transformation produces a set of coefficients that are equal
  // to the coefficients found by projecting the original function rotated by
  // the same rotation matrix.
  //
  // There are explicit instantiations for double, float, and Array3f.
  template <typename T>
  void Apply(const std::vector<T>& coeffs,  std::vector<T>* result) const;
  void Apply(const Eigen::MatrixXf& coeffs, Eigen::MatrixXf& result) const;
  void Apply(const Eigen::VectorXf& coeffs, Eigen::VectorXf& result) const;

  // The order (0-based) that the rotation was constructed with. It can only
  // transform coefficient vectors that were fit using the same order.
  int order() const;

  // Return the rotation that is effectively applied to the inputs of the
  // original function.
  Eigen::Quaternionf rotation() const;

  // Return the (2l+1)x(2l+1) matrix for transforming the coefficients within
  // band @l by the rotation. @l must be at least 0 and less than or equal to
  // the order this rotation was initially constructed with.
  const Eigen::MatrixXf& band_rotation(int l) const;

  Rotation(int order, const Eigen::Quaternionf& rotation);

 private:
  const int order_;
  const Eigen::Quaternionf rotation_;

  std::vector<Eigen::MatrixXf> band_rotations_;
};


Rotation::Rotation(int order, const Eigen::Quaternionf& rotation)
    : order_(order), rotation_(rotation) {
  band_rotations_.reserve(GetCoefficientCount(order));

  // Order 0 (first band) is simply the 1x1 identity since the SH basis
  // function is a simple sphere.
  Eigen::MatrixXf r(1, 1);
  r(0, 0) = 1.0;
  band_rotations_.push_back(r);

  r.resize(3, 3);
  // The second band's transformation is simply a permutation of the
  // rotation matrix's elements, provided in Appendix 1 of [1], updated to
  // include the Condon-Shortely phase. The recursive method in
  // ComputeBandRotation preserves the proper phases as high bands are computed.
  Eigen::Matrix3f rotation_mat = rotation.toRotationMatrix();
  r(0, 0) = rotation_mat(1, 1);
  r(0, 1) = -rotation_mat(1, 2);
  r(0, 2) = rotation_mat(1, 0);
  r(1, 0) = -rotation_mat(2, 1);
  r(1, 1) = rotation_mat(2, 2);
  r(1, 2) = -rotation_mat(2, 0);
  r(2, 0) = rotation_mat(0, 1);
  r(2, 1) = -rotation_mat(0, 2);
  r(2, 2) = rotation_mat(0, 0);
  band_rotations_.push_back(r);

  // Recursively build the remaining band rotations, using the equations
  // provided in [4, 4b].
  for (int l = 2; l <= order; l++) {
    ComputeBandRotation(l, &band_rotations_);
  }
}

/*
std::unique_ptr<Rotation> Rotation::Create(
    int order, const Eigen::Quaternionf& rotation) {
#ifndef NDEBUG
  CHECK(order >= 0, "Order must be at least 0.");
  CHECK(NearByMargin(rotation.squaredNorm(), 1.0),
        "Rotation must be normalized.");
#endif

  std::unique_ptr<Rotation> sh_rot(new Rotation(order, rotation));

  // Order 0 (first band) is simply the 1x1 identity since the SH basis
  // function is a simple sphere.
  Eigen::MatrixXf r(1, 1);
  r(0, 0) = 1.0;
  sh_rot->band_rotations_.push_back(r);

  r.resize(3, 3);
  // The second band's transformation is simply a permutation of the
  // rotation matrix's elements, provided in Appendix 1 of [1], updated to
  // include the Condon-Shortely phase. The recursive method in
  // ComputeBandRotation preserves the proper phases as high bands are computed.
  Eigen::Matrix3f rotation_mat = rotation.toRotationMatrix();
  r(0, 0) = rotation_mat(1, 1);
  r(0, 1) = -rotation_mat(1, 2);
  r(0, 2) = rotation_mat(1, 0);
  r(1, 0) = -rotation_mat(2, 1);
  r(1, 1) = rotation_mat(2, 2);
  r(1, 2) = -rotation_mat(2, 0);
  r(2, 0) = rotation_mat(0, 1);
  r(2, 1) = -rotation_mat(0, 2);
  r(2, 2) = rotation_mat(0, 0);
  sh_rot->band_rotations_.push_back(r);

  // Recursively build the remaining band rotations, using the equations
  // provided in [4, 4b].
  for (int l = 2; l <= order; l++) {
    ComputeBandRotation(l, &(sh_rot->band_rotations_));
  }

  return sh_rot;
}

std::unique_ptr<Rotation> Rotation::Create(int order,
                                           const Rotation& rotation) {
#ifndef NDEBUG
  CHECK(order >= 0, "Order must be at least 0.");
#endif

  std::unique_ptr<Rotation> sh_rot(new Rotation(order, rotation.rotation_));

  // Copy up to min(order, rotation.order_) band rotations into the new
  // SHRotation. For shared orders, they are the same. If the new order is
  // higher than already calculated then the remainder will be computed next.
  for (int l = 0; l <= std::min(order, rotation.order_); l++) {
    sh_rot->band_rotations_.push_back(rotation.band_rotations_[l]);
  }

  // Calculate remaining bands (automatically skipped if there are no more).
  for (int l = rotation.order_ + 1; l <= order; l++) {
    ComputeBandRotation(l, &(sh_rot->band_rotations_));
  }

  return sh_rot;
}
*/

int Rotation::order() const { return order_; }

Eigen::Quaternionf Rotation::rotation() const { return rotation_; }

const Eigen::MatrixXf& Rotation::band_rotation(int l) const {
  return band_rotations_[l];
}

template <typename T>
void Rotation::Apply(const std::vector<T>& coeff,
                     std::vector<T>* result) const {
#ifndef NDEBUG
  CHECK(coeff.size() == GetCoefficientCount(order_),
        "Incorrect number of coefficients provided.");
#endif

  // Resize to the required number of coefficients.
  // If result is already the same size as coeff, there's no need to zero out
  // its values since each index will be written explicitly later.
  if (result->size() != coeff.size()) {
    result->assign(coeff.size(), T());
  }

  // Because of orthogonality, the coefficients outside of each band do not
  // interact with one another. By separating them into band-specific matrices,
  // we take advantage of that sparsity.

  for (int l = 0; l <= order_; l++) {
    VectorX<T> band_coeff(2 * l + 1);

    // Fill band_coeff from the subset of @coeff that's relevant.
    for (int m = -l; m <= l; m++) {
      // Offset by l to get the appropiate vector component (0-based instead
      // of starting at -l).
      band_coeff(m + l) = coeff[GetIndex(l, m)];
    }

    band_coeff = band_rotations_[l].cast<T>() * band_coeff;

    // Copy rotated coefficients back into the appropriate subset into @result.
    for (int m = -l; m <= l; m++) {
      (*result)[GetIndex(l, m)] = band_coeff(m + l);
    }
  }
}

void Rotation::Apply(const Eigen::MatrixXf& coeffs,
                     Eigen::MatrixXf& result) const {
   const int rows = coeffs.cols();
   for(int l=0; l<=order_; ++l) {
      const int i = l*l;
      const int n = 2*l+1;
      result.block(i, 0, n, rows) = band_rotations_[l] * coeffs.block(i, 0, n, rows);
   }
}

void Rotation::Apply(const Eigen::VectorXf& coeffs,
                     Eigen::VectorXf& result) const {
   for(int l=0; l<=order_; ++l) {
      const int i = l*l;
      const int n = 2*l+1;
      result.segment(i, n) = band_rotations_[l] * coeffs.segment(i, n);
   }
}
