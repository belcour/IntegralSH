// STL includes
#include <iostream>

// Local includes
#include "SphericalIntegration.hpp"

/* Check the Lengendre expansion for the first elements in the matrix. This
 * test the init elements and the recursion pattern.
 */
int CheckLegendreExpansion(float Epsilon = 1.0E-5f) {

   Eigen::MatrixXf W = ZonalWeights(3);
   Eigen::MatrixXf R (3,3);

   const float factor = 1.0f/sqrt(4.0f*M_PI);
   R << 1.0f,       0.0f, -sqrt(5.0f/4.0f),
        0.0f, sqrt(3.0f),             0.0f,
        0.0f,       0.0f,      sqrt(45.0f);
   R *= factor;

   if(! W.isApprox(R, Epsilon)) {
      return 1;
   } else {
      return 0;
   }
}

int main(int argc, char** argv) {

   int nb_fails = 0;

   /* Check the basic elements of the ZH decomposition */
   nb_fails += CheckLegendreExpansion();

   if(nb_fails == 0) {
      return EXIT_SUCCESS;
   } else {
      return EXIT_FAILURE;
   }
}