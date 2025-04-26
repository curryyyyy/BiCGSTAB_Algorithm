#include "common.h"
#include <windows.h>  // For SetConsoleCtrlHandler
#include <omp.h>
#include <iostream>   // std::cout
#include <fstream>    // For file export
#include <iomanip>    // For std::setw, std::left, etc.

// Handler for console close events - ensures clean output before closing
BOOL WINAPI ConsoleHandler(DWORD dwCtrlType) {
    if (dwCtrlType == CTRL_CLOSE_EVENT) {
        std::cout << "Console is closing. Press Enter to continue..." << std::endl;
        std::cin.get();
    }
    return FALSE;
}

// Main function
int main(int argc, char* argv[]) {
    // Set up console close handler
    SetConsoleCtrlHandler(ConsoleHandler, TRUE);

    // Default settings
    bool runSerial = true;
    bool runParallel = true;
    bool runCuda = true;
    bool runHybridCuda = true;
    int num_threads = 16;
    std::string matrixFile = "C:\\Users\\Kelly\\Downloads\\Dataset\\cylshell_matrix_90mb.mtx";

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--serial-only") {
            runParallel = false;
            runCuda = false;
            runHybridCuda = false;
        }
        else if (arg == "--parallel-only") {
            runSerial = false;
            runCuda = false;
            runHybridCuda = false;
        }
        else if (arg == "--cuda-only") {
            runSerial = false;
            runParallel = false;
            runHybridCuda = false;
        }
        else if (arg == "--hybrid-only") {
            runSerial = false;
            runParallel = false;
            runCuda = false;
        }
        else if (arg == "--no-cuda") {
            runCuda = false;
            runHybridCuda = false;
        }
        else if (arg == "--no-hybrid") {
            runHybridCuda = false;
        }
        else if (arg == "--threads" && i + 1 < argc) {
            num_threads = std::atoi(argv[i + 1]);
            i++;
        }
        else if (arg == "--matrix" && i + 1 < argc) {
            matrixFile = argv[i + 1];
            i++;
        }
        else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                << "Options:\n"
                << "  --serial-only       Run only the serial version\n"
                << "  --parallel-only     Run only the parallel version\n"
                << "  --cuda-only         Run only the standard CUDA version\n"
                << "  --hybrid-only       Run only the hybrid (OpenMP+CUDA) version\n"
                << "  --no-cuda           Don't run the standard CUDA version\n"
                << "  --no-hybrid         Don't run the hybrid CUDA version\n"
                << "  --threads N         Set number of threads for parallel version (default: max available)\n"
                << "  --matrix FILE       Specify matrix market file to use (default: s3dkt3m2.mtx)\n"
                << "  --help              Display this help message\n";

            std::cout << "\nPress Enter to exit..." << std::endl;
            std::cin.get();
            return 0;
        }
    }

    // Set number of threads for OpenMP
#ifdef _OPENMP
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
        num_threads = omp_get_max_threads(); // Get actual number that was set
    }
    else {
        num_threads = omp_get_max_threads();
    }
    std::cout << "OpenMP is available. Maximum threads: " << num_threads << std::endl;
#else
    std::cout << "OpenMP is not available. Running in serial mode only." << std::endl;
    runParallel = false;
    // For hybrid approach, we need OpenMP too
    runHybridCuda = false;
#endif

    try {
        // Store all timing results in these variables
        struct TimingResults {
            double fileReadTime = 0.0;
            double csrConversionTime = 0.0;
            double solverTime = 0.0;
            double totalTime = 0.0;
            std::vector<double> solution;
        };

        TimingResults serialResults, openmpResults, cudaResults, hybridCudaResults;

        // Common matrix properties
        int matrixRows = 0, matrixCols = 0, matrixNnz = 0;

        // Read the matrix for all implementations (serial file reading for all)
        SparseMatrix sparseMatrix;

        // Serial file reading - used for ALL implementations
        std::cout << "Reading matrix file (serial): " << matrixFile << std::endl;
        auto startFileRead = std::chrono::high_resolution_clock::now();
        sparseMatrix = readMatrixMarket(matrixFile);
        auto endFileRead = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> durationFileRead = endFileRead - startFileRead;
        double fileReadTime = durationFileRead.count();
        std::cout << "Serial file reading time: " << fileReadTime << " ms\n";

        // Store matrix properties
        matrixRows = sparseMatrix.rows;
        matrixCols = sparseMatrix.cols;
        matrixNnz = sparseMatrix.nnz;
        sparseMatrix.printInfo();

        // Create the right-hand side vector (b) with all ones
        std::vector<double> b(matrixRows, 1.0);

        // Run serial implementation
        if (runSerial) {
            std::cout << "\n===== Running Serial Implementation =====\n";

            // Use the common file reading time
            serialResults.fileReadTime = fileReadTime;

            // CSR Conversion (serial)
            auto startCSR = std::chrono::high_resolution_clock::now();
            CSRMatrix csrMatrix(sparseMatrix, false);
            auto endCSR = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> durationCSR = endCSR - startCSR;
            serialResults.csrConversionTime = durationCSR.count();

            // Solver
            auto startSolve = std::chrono::high_resolution_clock::now();
            serialResults.solution = biCGSTAB_Serial(csrMatrix, b, 1e-8, 2000, true);
            auto endSolve = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> durationSolve = endSolve - startSolve;
            serialResults.solverTime = durationSolve.count();

            serialResults.totalTime = serialResults.fileReadTime +
                serialResults.csrConversionTime +
                serialResults.solverTime;

            std::cout << "Serial CSR conversion time: " << serialResults.csrConversionTime << " ms\n";
            std::cout << "Serial solver time: " << serialResults.solverTime << " ms\n";
            std::cout << "Serial total time: " << serialResults.totalTime << " ms\n";

            // Print solution samples
            std::cout << "Solution vector x (first 10 elements):" << std::endl;
            for (int i = 0; i < min(10, static_cast<int>(serialResults.solution.size())); ++i) {
                std::cout << "x[" << i << "] = " << serialResults.solution[i] << std::endl;
            }

            // Print solution norm for verification
            double norm = 0.0;
            for (const auto& val : serialResults.solution) {
                norm += val * val;
            }
            std::cout << "Serial solution norm: " << std::fixed << std::setprecision(8) << std::sqrt(norm) << std::endl;

            // Export solution vector to CSV
            std::ofstream serialFile("serial_solution.csv");
            if (serialFile.is_open()) {
                serialFile << "index,value\n";
                for (size_t i = 0; i < serialResults.solution.size(); i++) {
                    serialFile << i << "," << serialResults.solution[i] << "\n";
                }
                serialFile.close();
                std::cout << "Serial solution exported to serial_solution.csv" << std::endl;
            }

            // Extract dataset name for file naming
            std::string matrixFileName = matrixFile.substr(matrixFile.find_last_of("/\\") + 1);
            std::string datasetName = matrixFileName.substr(0, matrixFileName.find_last_of("."));

            // Export performance data to CSV with dataset in filename
            std::ofstream serialPerfFile(datasetName + "_serial_performance.csv");
            if (serialPerfFile.is_open()) {
                serialPerfFile << "method,matrix_size,solver_time_ms\n";
                serialPerfFile << "serial," << matrixRows << "," << serialResults.solverTime << "\n";
                serialPerfFile.close();
                std::cout << "Serial performance data exported to " << datasetName << "_serial_performance.csv" << std::endl;
            }
        }

        // Run OpenMP implementation
        if (runParallel) {
            std::cout << "\n===== Running OpenMP Implementation =====\n";

            // Use the common file reading time
            openmpResults.fileReadTime = fileReadTime;

            // CSR Conversion with OpenMP
            auto startCSR = std::chrono::high_resolution_clock::now();
            CSRMatrix csrMatrix(sparseMatrix, true);
            auto endCSR = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> durationCSR = endCSR - startCSR;
            openmpResults.csrConversionTime = durationCSR.count();

            // Solver with OpenMP
            auto startSolve = std::chrono::high_resolution_clock::now();
            openmpResults.solution = biCGSTAB_Parallel(csrMatrix, b, 1e-8, 2000, true);
            auto endSolve = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> durationSolve = endSolve - startSolve;
            openmpResults.solverTime = durationSolve.count();

            openmpResults.totalTime = openmpResults.fileReadTime +
                openmpResults.csrConversionTime +
                openmpResults.solverTime;

            std::cout << "OpenMP CSR conversion time: " << openmpResults.csrConversionTime << " ms\n";
            std::cout << "OpenMP solver time: " << openmpResults.solverTime << " ms\n";
            std::cout << "OpenMP total time: " << openmpResults.totalTime << " ms\n";

            // Print solution samples
            std::cout << "Solution vector x (first 10 elements):" << std::endl;
            for (int i = 0; i < min(10, static_cast<int>(openmpResults.solution.size())); ++i) {
                std::cout << "x[" << i << "] = " << openmpResults.solution[i] << std::endl;
            }

            // Print solution norm for verification
            double norm = 0.0;
#pragma omp parallel for reduction(+:norm)
            for (int i = 0; i < static_cast<int>(openmpResults.solution.size()); i++) {
                norm += openmpResults.solution[i] * openmpResults.solution[i];
            }
            std::cout << "OpenMP solution norm: " << std::fixed << std::setprecision(8) << std::sqrt(norm) << std::endl;

            // Export solution vector to CSV
            std::ofstream parallelFile("parallel_solution.csv");
            if (parallelFile.is_open()) {
                parallelFile << "index,value\n";
                for (size_t i = 0; i < openmpResults.solution.size(); i++) {
                    parallelFile << i << "," << openmpResults.solution[i] << "\n";
                }
                parallelFile.close();
                std::cout << "OpenMP solution exported to parallel_solution.csv" << std::endl;
            }

            // Extract dataset name for file naming
            std::string matrixFileName = matrixFile.substr(matrixFile.find_last_of("/\\") + 1);
            std::string datasetName = matrixFileName.substr(0, matrixFileName.find_last_of("."));

            // Export performance data to CSV with dataset in filename
            std::ofstream openmpPerfFile(datasetName + "_openmp_performance.csv");
            if (openmpPerfFile.is_open()) {
                openmpPerfFile << "method,matrix_size,solver_time_ms\n";
                openmpPerfFile << "openmp," << matrixRows << "," << openmpResults.solverTime << "\n";
                openmpPerfFile.close();
                std::cout << "OpenMP performance data exported to " << datasetName << "_openmp_performance.csv" << std::endl;
            }
        }

        // Run original CUDA implementation
        if (runCuda) {
            std::cout << "\n===== Running Standard CUDA Implementation =====\n";

            // Use the common file reading time
            cudaResults.fileReadTime = fileReadTime;

            // CSR Conversion with serial
            auto startCSR = std::chrono::high_resolution_clock::now();
            CSRMatrix csrMatrix(sparseMatrix, false); // Use serial conversion for standard CUDA
            auto endCSR = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> durationCSR = endCSR - startCSR;
            cudaResults.csrConversionTime = durationCSR.count();

            // Solver with CUDA
            auto startSolve = std::chrono::high_resolution_clock::now();
            cudaResults.solution = biCGSTAB_CUDA(csrMatrix, b, 1e-8, 2000, true);
            auto endSolve = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> durationSolve = endSolve - startSolve;
            cudaResults.solverTime = durationSolve.count();

            cudaResults.totalTime = cudaResults.fileReadTime +
                cudaResults.csrConversionTime +
                cudaResults.solverTime;

            std::cout << "Standard CUDA CSR conversion time: " << cudaResults.csrConversionTime << " ms\n";
            std::cout << "Standard CUDA solver time: " << cudaResults.solverTime << " ms\n";
            std::cout << "Standard CUDA total time: " << cudaResults.totalTime << " ms\n";

            // Print solution samples
            std::cout << "Solution vector x (first 10 elements):" << std::endl;
            for (int i = 0; i < min(10, static_cast<int>(cudaResults.solution.size())); ++i) {
                std::cout << "x[" << i << "] = " << cudaResults.solution[i] << std::endl;
            }

            // Print solution norm for verification
            double norm = 0.0;
            for (const auto& val : cudaResults.solution) {
                norm += val * val;
            }
            std::cout << "Standard CUDA solution norm: " << std::fixed << std::setprecision(8) << std::sqrt(norm) << std::endl;

            // Export solution vector to CSV
            std::ofstream cudaFile("cuda_standard_solution.csv");
            if (cudaFile.is_open()) {
                cudaFile << "index,value\n";
                for (size_t i = 0; i < cudaResults.solution.size(); i++) {
                    cudaFile << i << "," << cudaResults.solution[i] << "\n";
                }
                cudaFile.close();
                std::cout << "Standard CUDA solution exported to cuda_standard_solution.csv" << std::endl;
            }

            // Extract dataset name for file naming
            std::string matrixFileName = matrixFile.substr(matrixFile.find_last_of("/\\") + 1);
            std::string datasetName = matrixFileName.substr(0, matrixFileName.find_last_of("."));

            // Export performance data to CSV with dataset in filename
            std::ofstream stdCudaPerfFile(datasetName + "_standard_cuda_performance.csv");
            if (stdCudaPerfFile.is_open()) {
                stdCudaPerfFile << "method,matrix_size,solver_time_ms\n";
                stdCudaPerfFile << "standard_cuda," << matrixRows << "," << cudaResults.solverTime << "\n";
                stdCudaPerfFile.close();
                std::cout << "Standard CUDA performance data exported to " << datasetName << "_standard_cuda_performance.csv" << std::endl;
            }
        }

        // Run hybrid CUDA implementation (OpenMP for CSR conversion + CUDA for solving)
        if (runHybridCuda) {
            std::cout << "\n===== Running Hybrid CUDA Implementation (OpenMP+CUDA) =====\n";

            // Use the common serial file reading time for hybrid CUDA
            hybridCudaResults.fileReadTime = fileReadTime;

            // CSR Conversion with OpenMP
            auto startCSR = std::chrono::high_resolution_clock::now();
            CSRMatrix csrMatrix(sparseMatrix, true); // Use parallel conversion
            auto endCSR = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> durationCSR = endCSR - startCSR;
            hybridCudaResults.csrConversionTime = durationCSR.count();

            // Solver with CUDA
            auto startSolve = std::chrono::high_resolution_clock::now();
            hybridCudaResults.solution = biCGSTAB_CUDA(csrMatrix, b, 1e-8, 2000, true);
            auto endSolve = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> durationSolve = endSolve - startSolve;
            hybridCudaResults.solverTime = durationSolve.count();

            hybridCudaResults.totalTime = hybridCudaResults.fileReadTime +
                hybridCudaResults.csrConversionTime +
                hybridCudaResults.solverTime;

            std::cout << "Hybrid CUDA file reading time (serial): " << hybridCudaResults.fileReadTime << " ms\n";
            std::cout << "Hybrid CUDA CSR conversion time (OpenMP): " << hybridCudaResults.csrConversionTime << " ms using " << num_threads << " threads\n";
            std::cout << "Hybrid CUDA solver time: " << hybridCudaResults.solverTime << " ms\n";
            std::cout << "Hybrid CUDA total time: " << hybridCudaResults.totalTime << " ms\n";

            // Print solution samples
            std::cout << "Solution vector x (first 10 elements):" << std::endl;
            for (int i = 0; i < min(10, static_cast<int>(hybridCudaResults.solution.size())); ++i) {
                std::cout << "x[" << i << "] = " << hybridCudaResults.solution[i] << std::endl;
            }

            // Print solution norm for verification
            double norm = 0.0;
            for (const auto& val : hybridCudaResults.solution) {
                norm += val * val;
            }
            std::cout << "Hybrid CUDA solution norm: " << std::fixed << std::setprecision(8) << std::sqrt(norm) << std::endl;

            // Export solution vector to CSV
            std::ofstream hybridFile("cuda_hybrid_solution.csv");
            if (hybridFile.is_open()) {
                hybridFile << "index,value\n";
                for (size_t i = 0; i < hybridCudaResults.solution.size(); i++) {
                    hybridFile << i << "," << hybridCudaResults.solution[i] << "\n";
                }
                hybridFile.close();
                std::cout << "Hybrid CUDA solution exported to cuda_hybrid_solution.csv" << std::endl;
            }

            // Extract dataset name for file naming
            std::string matrixFileName = matrixFile.substr(matrixFile.find_last_of("/\\") + 1);
            std::string datasetName = matrixFileName.substr(0, matrixFileName.find_last_of("."));

            // Export performance data to CSV with dataset in filename
            std::ofstream hybridCudaPerfFile(datasetName + "_hybrid_cuda_performance.csv");
            if (hybridCudaPerfFile.is_open()) {
                hybridCudaPerfFile << "method,matrix_size,solver_time_ms\n";
                hybridCudaPerfFile << "hybrid_cuda," << matrixRows << "," << hybridCudaResults.solverTime << "\n";
                hybridCudaPerfFile.close();
                std::cout << "Hybrid CUDA performance data exported to " << datasetName << "_hybrid_cuda_performance.csv" << std::endl;
            }
        }

        // ========== Performance Comparison ==========
        std::cout << "\n===== Performance Comparison =====\n";

        // Create a nicely formatted table
        const int colWidth = 15;

        // Table header
        std::cout << std::left << std::setw(25) << "Implementation"
            << std::setw(colWidth) << "File Read (ms)"
            << std::setw(colWidth) << "CSR Conv. (ms)"
            << std::setw(colWidth) << "Solver (ms)"
            << std::setw(colWidth) << "Total (ms)" << std::endl;

        std::cout << std::string(25 + colWidth * 4, '-') << std::endl;

        // Table rows
        if (runSerial) {
            std::cout << std::left << std::setw(25) << "Serial"
                << std::setw(colWidth) << serialResults.fileReadTime
                << std::setw(colWidth) << serialResults.csrConversionTime
                << std::setw(colWidth) << serialResults.solverTime
                << std::setw(colWidth) << serialResults.totalTime << std::endl;
        }

        if (runParallel) {
            std::cout << std::left << std::setw(25) << "OpenMP"
                << std::setw(colWidth) << openmpResults.fileReadTime
                << std::setw(colWidth) << openmpResults.csrConversionTime
                << std::setw(colWidth) << openmpResults.solverTime
                << std::setw(colWidth) << openmpResults.totalTime << std::endl;
        }

        if (runCuda) {
            std::cout << std::left << std::setw(25) << "Standard CUDA"
                << std::setw(colWidth) << cudaResults.fileReadTime
                << std::setw(colWidth) << cudaResults.csrConversionTime
                << std::setw(colWidth) << cudaResults.solverTime
                << std::setw(colWidth) << cudaResults.totalTime << std::endl;
        }

        if (runHybridCuda) {
            std::cout << std::left << std::setw(25) << "Hybrid CUDA (OpenMP+CUDA)"
                << std::setw(colWidth) << hybridCudaResults.fileReadTime
                << std::setw(colWidth) << hybridCudaResults.csrConversionTime
                << std::setw(colWidth) << hybridCudaResults.solverTime
                << std::setw(colWidth) << hybridCudaResults.totalTime << std::endl;
        }

        std::cout << std::endl;

        // Calculate and display speedups
        std::cout << "===== Speedup Analysis =====\n";

        // Only show speedups if serial is run (as the baseline)
        if (runSerial) {
            // Table header for speedups
            std::cout << std::left << std::setw(25) << "Comparison"
                << std::setw(colWidth) << "File Read"
                << std::setw(colWidth) << "CSR Conv."
                << std::setw(colWidth) << "Solver"
                << std::setw(colWidth) << "Total" << std::endl;

            std::cout << std::string(25 + colWidth * 4, '-') << std::endl;

            // OpenMP speedup
            if (runParallel) {
                double fileReadSpeedup = serialResults.fileReadTime / openmpResults.fileReadTime;
                double csrSpeedup = serialResults.csrConversionTime / openmpResults.csrConversionTime;
                double solverSpeedup = serialResults.solverTime / openmpResults.solverTime;
                double totalSpeedup = serialResults.totalTime / openmpResults.totalTime;

                std::cout << std::left << std::setw(25) << "Serial vs OpenMP"
                    << std::setw(colWidth) << fileReadSpeedup
                    << std::setw(colWidth) << csrSpeedup
                    << std::setw(colWidth) << solverSpeedup
                    << std::setw(colWidth) << totalSpeedup << std::endl;

                // OpenMP efficiency
                double efficiency = (solverSpeedup / num_threads) * 100;
                std::cout << "OpenMP Efficiency: " << efficiency << "%" << std::endl;
            }

            // Standard CUDA speedup
            if (runCuda) {
                double fileReadSpeedup = serialResults.fileReadTime / cudaResults.fileReadTime;
                double csrSpeedup = serialResults.csrConversionTime / cudaResults.csrConversionTime;
                double solverSpeedup = serialResults.solverTime / cudaResults.solverTime;
                double totalSpeedup = serialResults.totalTime / cudaResults.totalTime;

                std::cout << std::left << std::setw(25) << "Serial vs Standard CUDA"
                    << std::setw(colWidth) << fileReadSpeedup
                    << std::setw(colWidth) << csrSpeedup
                    << std::setw(colWidth) << solverSpeedup
                    << std::setw(colWidth) << totalSpeedup << std::endl;
            }

            // Hybrid CUDA speedup
            if (runHybridCuda) {
                double fileReadSpeedup = serialResults.fileReadTime / hybridCudaResults.fileReadTime;
                double csrSpeedup = serialResults.csrConversionTime / hybridCudaResults.csrConversionTime;
                double solverSpeedup = serialResults.solverTime / hybridCudaResults.solverTime;
                double totalSpeedup = serialResults.totalTime / hybridCudaResults.totalTime;

                std::cout << std::left << std::setw(25) << "Serial vs Hybrid CUDA"
                    << std::setw(colWidth) << fileReadSpeedup
                    << std::setw(colWidth) << csrSpeedup
                    << std::setw(colWidth) << solverSpeedup
                    << std::setw(colWidth) << totalSpeedup << std::endl;
            }

            std::cout << std::endl;
        }

        // Compare OpenMP and CUDA implementations
        if (runParallel && runCuda) {
            double fileReadSpeedup = openmpResults.fileReadTime / cudaResults.fileReadTime;
            double csrSpeedup = openmpResults.csrConversionTime / cudaResults.csrConversionTime;
            double solverSpeedup = openmpResults.solverTime / cudaResults.solverTime;
            double totalSpeedup = openmpResults.totalTime / cudaResults.totalTime;

            std::cout << std::left << std::setw(25) << "OpenMP vs Standard CUDA"
                << std::setw(colWidth) << fileReadSpeedup
                << std::setw(colWidth) << csrSpeedup
                << std::setw(colWidth) << solverSpeedup
                << std::setw(colWidth) << totalSpeedup << std::endl;
        }

        // Compare OpenMP and Hybrid CUDA
        if (runParallel && runHybridCuda) {
            double fileReadSpeedup = openmpResults.fileReadTime / hybridCudaResults.fileReadTime;
            double csrSpeedup = openmpResults.csrConversionTime / hybridCudaResults.csrConversionTime;
            double solverSpeedup = openmpResults.solverTime / hybridCudaResults.solverTime;
            double totalSpeedup = openmpResults.totalTime / hybridCudaResults.totalTime;

            std::cout << std::left << std::setw(25) << "OpenMP vs Hybrid CUDA"
                << std::setw(colWidth) << fileReadSpeedup
                << std::setw(colWidth) << csrSpeedup
                << std::setw(colWidth) << solverSpeedup
                << std::setw(colWidth) << totalSpeedup << std::endl;
        }

        // Compare Standard CUDA and Hybrid CUDA
        if (runCuda && runHybridCuda) {
            double fileReadSpeedup = cudaResults.fileReadTime / hybridCudaResults.fileReadTime;
            double csrSpeedup = cudaResults.csrConversionTime / hybridCudaResults.csrConversionTime;
            double solverSpeedup = cudaResults.solverTime / hybridCudaResults.solverTime;
            double totalSpeedup = cudaResults.totalTime / hybridCudaResults.totalTime;

            std::cout << std::left << std::setw(25) << "Standard vs Hybrid CUDA"
                << std::setw(colWidth) << fileReadSpeedup
                << std::setw(colWidth) << csrSpeedup
                << std::setw(colWidth) << solverSpeedup
                << std::setw(colWidth) << totalSpeedup << std::endl;
        }

        // Solution verification
        std::cout << "\n===== Solution Verification =====\n";

        // Verify accuracy between implementations
        if (runSerial && runParallel) {
            double diffNorm = 0.0;
            for (size_t i = 0; i < serialResults.solution.size(); i++) {
                double diff = serialResults.solution[i] - openmpResults.solution[i];
                diffNorm += diff * diff;
            }
            diffNorm = std::sqrt(diffNorm);

            std::cout << "Serial vs OpenMP solution difference norm: " << std::fixed
                << std::setprecision(8) << diffNorm << std::endl;

            if (diffNorm < 1e-6) {
                std::cout << "Serial and OpenMP solutions match (within tolerance).\n";
            }
            else {
                std::cout << "WARNING: Solutions differ between Serial and OpenMP versions!\n";
            }
        }

        if (runSerial && runCuda) {
            double diffNorm = 0.0;
            for (size_t i = 0; i < serialResults.solution.size(); i++) {
                double diff = serialResults.solution[i] - cudaResults.solution[i];
                diffNorm += diff * diff;
            }
            diffNorm = std::sqrt(diffNorm);

            std::cout << "Serial vs Standard CUDA solution difference norm: " << std::fixed
                << std::setprecision(8) << diffNorm << std::endl;

            if (diffNorm < 1e-6) {
                std::cout << "Serial and Standard CUDA solutions match (within tolerance).\n";
            }
            else {
                std::cout << "WARNING: Solutions differ between Serial and Standard CUDA versions!\n";
            }
        }

        if (runSerial && runHybridCuda) {
            double diffNorm = 0.0;
            for (size_t i = 0; i < serialResults.solution.size(); i++) {
                double diff = serialResults.solution[i] - hybridCudaResults.solution[i];
                diffNorm += diff * diff;
            }
            diffNorm = std::sqrt(diffNorm);

            std::cout << "Serial vs Hybrid CUDA solution difference norm: " << std::fixed
                << std::setprecision(8) << diffNorm << std::endl;

            if (diffNorm < 1e-6) {
                std::cout << "Serial and Hybrid CUDA solutions match (within tolerance).\n";
            }
            else {
                std::cout << "WARNING: Solutions differ between Serial and Hybrid CUDA versions!\n";
            }
        }

        if (runParallel && runCuda) {
            double diffNorm = 0.0;
            for (size_t i = 0; i < openmpResults.solution.size(); i++) {
                double diff = openmpResults.solution[i] - cudaResults.solution[i];
                diffNorm += diff * diff;
            }
            diffNorm = std::sqrt(diffNorm);

            std::cout << "OpenMP vs Standard CUDA solution difference norm: " << std::fixed
                << std::setprecision(8) << diffNorm << std::endl;

            if (diffNorm < 1e-6) {
                std::cout << "OpenMP and Standard CUDA solutions match (within tolerance).\n";
            }
            else {
                std::cout << "WARNING: Solutions differ between OpenMP and Standard CUDA versions!\n";
            }
        }

        if (runParallel && runHybridCuda) {
            double diffNorm = 0.0;
            for (size_t i = 0; i < openmpResults.solution.size(); i++) {
                double diff = openmpResults.solution[i] - hybridCudaResults.solution[i];
                diffNorm += diff * diff;
            }
            diffNorm = std::sqrt(diffNorm);

            std::cout << "OpenMP vs Hybrid CUDA solution difference norm: " << std::fixed
                << std::setprecision(8) << diffNorm << std::endl;

            if (diffNorm < 1e-6) {
                std::cout << "OpenMP and Hybrid CUDA solutions match (within tolerance).\n";
            }
            else {
                std::cout << "WARNING: Solutions differ between OpenMP and Hybrid CUDA versions!\n";
            }
        }

        if (runCuda && runHybridCuda) {
            double diffNorm = 0.0;
            for (size_t i = 0; i < cudaResults.solution.size(); i++) {
                double diff = cudaResults.solution[i] - hybridCudaResults.solution[i];
                diffNorm += diff * diff;
            }
            diffNorm = std::sqrt(diffNorm);

            std::cout << "Standard CUDA vs Hybrid CUDA solution difference norm: " << std::fixed
                << std::setprecision(8) << diffNorm << std::endl;

            if (diffNorm < 1e-6) {
                std::cout << "Standard CUDA and Hybrid CUDA solutions match (within tolerance).\n";
            }
            else {
                std::cout << "WARNING: Solutions differ between Standard CUDA and Hybrid CUDA versions!\n";
            }
        }

        // Extract dataset name for comprehensive CSV file
        std::string matrixFileName = matrixFile.substr(matrixFile.find_last_of("/\\") + 1);
        std::string datasetName = matrixFileName.substr(0, matrixFileName.find_last_of("."));

        // Export ALL performance data to a single combined CSV file
        std::ofstream perfFile(datasetName + "_performance.csv");
        if (perfFile.is_open()) {
            // Write header with the requested column names
            perfFile << "method,matrix_size,solver_time_ms\n";

            // Matrix size
            int matrix_size = matrixRows;

            // Write data for each implementation that was run (solver time only)
            if (runSerial) {
                perfFile << "serial," << matrix_size << "," << serialResults.solverTime << "\n";
            }

            if (runParallel) {
                perfFile << "openmp," << matrix_size << "," << openmpResults.solverTime << "\n";
            }

            if (runCuda) {
                perfFile << "standard_cuda," << matrix_size << "," << cudaResults.solverTime << "\n";
            }

            if (runHybridCuda) {
                perfFile << "hybrid_cuda," << matrix_size << "," << hybridCudaResults.solverTime << "\n";
            }

            perfFile.close();
            std::cout << "\nComprehensive performance data exported to " << datasetName << "_performance.csv" << std::endl;
        }

    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    catch (...) {
        std::cerr << "Unknown error occurred!" << std::endl;
    }

    // Keep console window open when running from Visual Studio
    std::cout << "\nPress Enter to exit..." << std::endl;
    std::cin.get();

    return 0;
}