// STL includes
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <utility>
#include <chrono>
typedef std::chrono::high_resolution_clock Clock;

// Local includes
#include "Tests.hpp"
#include "SH.hpp"
#include "SphericalHarmonics.hpp"
#include "SphericalIntegration.hpp"
#include "DirectionsSampling.hpp"

// GLM include
#include <glm/glm.hpp>

int main(int argc, char** argv) {

    auto A = glm::vec3( 0.0, 0.5, 0.5);
    auto B = glm::vec3(-0.5, 0.5, 0.0);
    auto C = glm::vec3( 0.5, 0.5, 0.0);
    auto tri = Triangle(glm::normalize(A), glm::normalize(B), glm::normalize(C));

    std::cout <<"#\to\tn\tms\tms" << std::endl;

    int max_order  = 18;
    int max_trials = 1000;
    for(int order=1; order<max_order; ++order) {

        // Generate the coefficients, use the zero function
        const int nb_coeffs = SHTerms(order);
        Eigen::VectorXf clm = Eigen::VectorXf::Random(nb_coeffs);

        // Analytical solution
        const auto basis = SamplingBlueNoise<Vector>(2*order+1);
        
        // Get the Zonal weights matrix and the Zlm -> Ylm conversion matrix
        // and compute the product of the two: `Prod = A x Zw`.
        const auto ZW = ZonalWeights<Vector>(basis);
        const auto Y  = ZonalExpansion<SH, Vector>(basis);
        const auto A  = computeInverse(Y);

        const auto Prod = (A*ZW).eval();
        const auto Fact = (Prod.transpose() * clm).eval();

        // Analytical evaluation of the integral of power of cosines for
        // the different basis elements up to the order defined by the
        // number of elements in the basis
        auto start = Clock::now();
        float shI  = 0.0;
        for(int trial=0; trial<max_trials; ++trial) {
            const auto moments = AxialMoments<Triangle, Vector>(tri, basis);
            shI += moments[0];
        }
        shI /= max_trials;
        auto end = Clock::now();
        double timing_no_mult = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / (double)max_trials;

        start = Clock::now();
        shI  = 0.0;
        for(int trial=0; trial<max_trials; ++trial) {
            const auto moments = AxialMoments<Triangle, Vector>(tri, basis);
            //shI += Fact.dot(moments);
            shI += clm.dot(Prod * moments);
        }
        shI /= max_trials;
        end = Clock::now();
        double timing_mult = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / (double)max_trials;


        std::cout << "\t" << order << "\t" << (order+1)*(order+1) << "\t" << timing_no_mult  << "\t" << timing_mult << std::endl;
    }

    return EXIT_SUCCESS;
}
