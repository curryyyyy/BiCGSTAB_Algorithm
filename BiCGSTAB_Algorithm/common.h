#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <map>
#include <algorithm>
#include <numeric>
#include <string>

// Forward declarations for solver functions
struct SparseMatrix;
struct CSRMatrix;
std::vector<double> biCGSTAB_Serial(const CSRMatrix& A, const std::vector<double>& b,
    double tol = 1e-8, int maxIter = 1000, bool usePreconditioner = true);
std::vector<double> biCGSTAB_Parallel(const CSRMatrix& A, const std::vector<double>& b,
    double tol = 1e-8, int maxIter = 1000, bool usePreconditioner = true);

// Sparse matrix in COO format (Coordinate format)
struct SparseMatrix {
    int rows = 0, cols = 0, nnz = 0;
    std::vector<int> row_indices;
    std::vector<int> col_indices;
    std::vector<double> values;

    // Print matrix information
    void printInfo() const {
        std::cout << "Matrix dimensions: " << rows << " x " << cols << std::endl;
        std::cout << "Non-zero elements: " << nnz << std::endl;

        // Calculate memory usage
        size_t memoryUsage = sizeof(int) * 3;  // rows, cols, nnz
        memoryUsage += sizeof(int) * row_indices.size();
        memoryUsage += sizeof(int) * col_indices.size();
        memoryUsage += sizeof(double) * values.size();

        std::cout << "Sparse matrix memory usage: " << memoryUsage / (1024.0 * 1024.0) << " MB" << std::endl;

        // Calculate memory if converted to dense
        size_t denseMemory = sizeof(double) * rows * cols;
        std::cout << "If converted to dense: " << denseMemory / (1024.0 * 1024.0) << " MB" << std::endl;
    }
};

// Compressed Sparse Row format
struct CSRMatrix {
    int rows, cols, nnz;
    std::vector<int> row_ptr;
    std::vector<int> col_ind;
    std::vector<double> values;

    // Constructor based on mode - implemented inline to avoid incomplete type issues
    CSRMatrix(const SparseMatrix& mat, bool useParallel) {
        if (useParallel) {
            convertParallelCSR(mat);
        }
        else {
            convertSerialCSR(mat);
        }
    }

    // Serial methods
    void convertSerialCSR(const SparseMatrix& mat);
    std::vector<double> getDiagonalSerial() const;
    void multiplySerial(const std::vector<double>& x, std::vector<double>& y) const;

    // Parallel methods
    void convertParallelCSR(const SparseMatrix& mat);
    std::vector<double> getDiagonalParallel() const;
    void multiplyParallel(const std::vector<double>& x, std::vector<double>& y) const;
};

// Implementation of readMatrixMarket
inline SparseMatrix readMatrixMarket(const std::string& filename) {
    std::ifstream file(filename);
    SparseMatrix mat;

    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        std::cerr << "Make sure the file exists in the project directory or provide a full path." << std::endl;

        // In Windows, wait for user input before exiting
        std::cout << "Press Enter to exit..." << std::endl;
        std::cin.get();
        exit(1);
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line[0] != '%') break;
    }

    std::stringstream header(line);
    header >> mat.rows >> mat.cols >> mat.nnz;

    // Check if matrix size is reasonable before allocating memory
    size_t estimatedMemory = sizeof(double) * mat.rows * mat.cols;
    if (estimatedMemory > 1024 * 1024 * 1024) {  // More than 1 GB
        std::cout << "Warning: Converting to dense would require approximately "
            << (estimatedMemory / (1024.0 * 1024.0)) << " MB of memory." << std::endl;
        std::cout << "Do you want to continue? (y/n): ";
        char response;
        std::cin >> response;
        if (response != 'y' && response != 'Y') {
            std::cout << "Operation cancelled by user." << std::endl;

            // In Windows, wait for user input before exiting
            std::cout << "Press Enter to exit..." << std::endl;
            std::cin.get(); // Clear the newline
            std::cin.get();
            exit(0);
        }
    }

    mat.row_indices.resize(mat.nnz);
    mat.col_indices.resize(mat.nnz);
    mat.values.resize(mat.nnz);

    for (int i = 0; i < mat.nnz; ++i) {
        int r, c;
        double val;
        file >> r >> c >> val;
        mat.row_indices[i] = r - 1; // Matrix Market is 1-based
        mat.col_indices[i] = c - 1;
        mat.values[i] = val;
    }

    return mat;
}