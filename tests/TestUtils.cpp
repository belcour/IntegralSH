// Eigen includes
#include <Eigen/Core>

// STL includes
#include <iostream>
#include <vector>

// Library include
#include <Utils.hpp>
#include "Tests.hpp"


/* Check if clamping a polyong works */
int CheckPolygonClamping() {

   int nb_fails = 0;
   PolygonConstructor pConstruct;
   Polygon polygon;
   Vector A, B, C;

   // Create a polygon above the horizon
   A = Vector(0,0,1);
   B = Vector(0,1,1);
   C = Vector(1,0,1);
   pConstruct = PolygonConstructor(Vector::Normalize(A), Vector::Normalize(B), Vector::Normalize(C));
   polygon = pConstruct.ProjectToHemisphere(Vector(0,0,0), Vector(0,0,1));
   if(polygon.size() != 3) {
      std::cerr << "Error: projecting canonical triangle doesn't provide the same triangle" << std::endl;
      std::cout << polygon << std::endl;
      ++nb_fails;
   }

   // Create a polygon above the horizon
   A = Vector(0,0,1);
   B = Vector(0,1,0);
   C = Vector(1,0,0);
   pConstruct = PolygonConstructor(A, B, C);
   polygon = pConstruct.ProjectToHemisphere(Vector(0,0,0), Vector(0,0,1));
   if(polygon.size() != 3) {
      std::cerr << "Error: projecting canonical triangle doesn't provide the same triangle" << std::endl;
      std::cout << polygon << std::endl;
      ++nb_fails;
   }

   // Create a polygon below the horizon
   A = Vector(0,0,1);
   B = Vector(0,1,1);
   C = Vector(1,0,1);
   pConstruct = PolygonConstructor(Vector::Normalize(A), Vector::Normalize(B), Vector::Normalize(C));
   polygon = pConstruct.ProjectToHemisphere(Vector(0,0,0), Vector(0,0,-1));
   if(polygon.size() != 0) {
      std::cerr << "Error: projecting canonical triangle should return nothing" << std::endl;
      std::cout << polygon << std::endl;
      ++nb_fails;
   }

   // Create a polygon crossing the horizon
   A = Vector(0, 0, 1);
   B = Vector(0, 1, 1);
   C = Vector(1, 0,-1);
   pConstruct = PolygonConstructor(Vector::Normalize(A), Vector::Normalize(B), Vector::Normalize(C));
   polygon = pConstruct.ProjectToHemisphere(Vector(0,0,0), Vector(0,0,1));
   if(polygon.size() != 4) {
      std::cerr << "Error: projecting canonical crossing triangle should return a 4-polygon" << std::endl;
      std::cout << polygon << std::endl;
      ++nb_fails;
   }

   // Create a polygon crossing the horizon
   A = Vector(0, 0, 1);
   B = Vector(0, 1, 0);
   C = Vector(1, 0,-1);
   pConstruct = PolygonConstructor(Vector::Normalize(A), Vector::Normalize(B), Vector::Normalize(C));
   polygon = pConstruct.ProjectToHemisphere(Vector(0,0,0), Vector(0,0,1));
   if(polygon.size() != 3) {
      std::cerr << "Error: projecting canonical crossing triangle should return a 3-polygon" << std::endl;
      std::cout << polygon << std::endl;
      ++nb_fails;
   }


   // Create a polygon crossing the horizon
   A = Vector(0, 0, 1);
   B = Vector(0, 1, 0);
   C = Vector(1, 0,-1);
   pConstruct = PolygonConstructor(Vector::Normalize(A), Vector::Normalize(B), Vector::Normalize(C));
   pConstruct.push_back(Vector::Normalize(Vector(0,0,-1)));
   pConstruct.push_back(Vector::Normalize(Vector(1,0,-1)));
   polygon = pConstruct.ProjectToHemisphere(Vector(0,0,0), Vector(0,0,1));
   if(polygon.size() != 3) {
      std::cerr << "Error: projecting canonical crossing triangle should return a 3-polygon" << std::endl;
      std::cout << polygon << std::endl;
      ++nb_fails;
   }
   return nb_fails;
}

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


   /* Test the polygon projection code */
   nb_fails += CheckPolygonClamping();


   if(nb_fails > 0) {
      std::cerr << "Failure!" << std::endl;
      return EXIT_FAILURE;
   } else {
      std::cerr << "Success!" << std::endl;
      return EXIT_SUCCESS;
   }
}
