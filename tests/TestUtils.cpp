// Eigen includes
#include <Eigen/Core>

// STL includes
#include <iostream>
#include <vector>

// Library include
#include <Utils.hpp>

int main(int argc, char** argv) {

   int nmats = 5;
   int ncols = 10;
   int nrows = 15;

   // Create matrices
   std::vector<Eigen::MatrixXf> matrices;
   for(int n=0; n<nmats; ++n) {
      Eigen::MatrixXf matrix = Eigen::MatrixXf::Random(ncols, nrows);
      matrices.push_back(matrix);
   }

   // Save the matrices
   SaveMatrices("test.mats", matrices);

   // Load them and compare
   std::vector<Eigen::MatrixXf> read_mats = LoadMatrices("test.mats");

   int nb_fails = 0;
   for(int i=0; i<nmats; ++i) {
      if(!matrices[i].isApprox(read_mats[i])) {
         ++nb_fails;
      }
   }

   try {
      auto test = LoadMatrices("inexistant_file_on_disk.mats");
   } catch(IOError& error) {
      std::cerr << "Correct: unable to open file (testing exceptions)" << std::endl;
      std::cerr << error.what() << std::endl;
   }

   if(nb_fails > 0) {
      std::cerr << "Failure, did not export the matrices correctly." << std::endl;
      return EXIT_FAILURE;
   } else {
      std::cerr << "Success !" << std::endl;
      return EXIT_SUCCESS;
   }
}
