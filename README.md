## Analytical Integration of Spherical Harmonics ##

This code implements the analytical integration method from *A Closed-Form Method for Integrating Arbitrary Spherical Polynomials*.

### Use this code ###

The integration code is header only and template based. It requires to compile with a C++11 compatible compiler and uses [Eigen](http://eigen.tuxfamily.org) to perform the required linear algebra operations. Eigen is defined as a submodule of this git repository.

A CMake script is provided to perform the sanity check of the code. Examples of use of the spherical integration code can be found in `test`.