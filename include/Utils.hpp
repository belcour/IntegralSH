#pragma once

// Include Eigen
#include <Eigen/Core>

// Include STL
#include <fstream>
#include <string>
#include <vector>
#include <exception>

struct IOError : public std::exception {
   std::string _msg;

   IOError(const std::string& filename = "") {
      _msg = "Unable to open file descriptor \"";
      _msg.append(filename);
      _msg.append("\"");
   }

   virtual const char* what() const throw() {
      return _msg.c_str();
   }
};

/* Enable to load a list of Eigen Matrices to binary format
 */
inline std::vector<Eigen::MatrixXf> LoadMatrices(const std::string& filename) {
   // Open file
   auto out = std::fopen(filename.c_str(), "r");
   if(out == nullptr) {
      IOError ioException(filename);
      throw ioException;
   }

   // Load the number of matrices
   int size;
   std::fread(&size, sizeof(int), 1, out);
   std::vector<Eigen::MatrixXf> matrices(size);

   for(int k=0; k<size; ++k) {
      Eigen::MatrixXf& mat = matrices[k];

      int nrows;
      int ncols;
      size_t s = std::fread(&nrows, sizeof(int), 1, out);
      assert(s == 1);
      s = std::fread(&ncols, sizeof(int), 1, out);
      assert(s == 1);
      mat = Eigen::MatrixXf(nrows, ncols);
      s = std::fread(mat.data(), sizeof(float), nrows*ncols, out);
      assert(s == size_t(nrows*ncols));
   }
   std::fclose(out);

   return matrices;
}

/* Enable to save a list of Eigen Matrices from a binary format
 */
inline void SaveMatrices(const std::string& filename,
                         const std::vector<Eigen::MatrixXf>& matrices) {

   // File descriptor
   auto out = std::fopen(filename.c_str(), "w");

   // Output list of matrices
   const int size = matrices.size();
   std::fwrite(&size, sizeof(int), 1, out);
   for(const auto& mat : matrices) {
      // Output the matrix size
      const int nrows = mat.rows();
      const int ncols =  mat.cols();
      size_t s = std::fwrite(&nrows, sizeof(int), 1, out);
      assert(s == 1);
      s = std::fwrite(&ncols, sizeof(int), 1, out);
      assert(s == 1);
      s = std::fwrite(mat.data(), sizeof(float), nrows*ncols, out);
      assert(s == size_t(nrows*ncols));
   }
   std::fclose(out);
}

/* Enable to load a list of Eigen Matrices to binary format
 */
inline Eigen::MatrixXf LoadMatrix(const std::string& filename) {
   // Open file
   auto out = std::fopen(filename.c_str(), "r");
   if(out == nullptr) {
      IOError ioException(filename);
      throw ioException;
   }

   // Load the number of matrices
   Eigen::MatrixXf mat;
   int nrows;
   int ncols;
   size_t s = std::fread(&nrows, sizeof(int), 1, out);
   assert(s == 1);
   s = std::fread(&ncols, sizeof(int), 1, out);
   assert(s == 1);
   mat = Eigen::MatrixXf(nrows, ncols);
   s = std::fread(mat.data(), sizeof(float), nrows*ncols, out);
   assert(s == size_t(nrows*ncols));

   std::fclose(out);
   return mat;
}

/* Enable to save a list of Eigen Matrices from a binary format
 */
inline void SaveMatrix(const std::string& filename,
                         const Eigen::MatrixXf& mat) {

   // File descriptor
   auto out = std::fopen(filename.c_str(), "w");

   // Output the matrix size
   const int nrows = mat.rows();
   const int ncols = mat.cols();
   size_t s = std::fwrite(&nrows, sizeof(int), 1, out);
   assert(s == 1);
   s = std::fwrite(&ncols, sizeof(int), 1, out);
   assert(s == 1);
   s = std::fwrite(mat.data(), sizeof(float), nrows*ncols, out);
   assert(s == size_t(nrows*ncols));

   std::fclose(out);
}