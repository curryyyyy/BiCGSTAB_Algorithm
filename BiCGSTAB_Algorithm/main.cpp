#include "common.h"
#include <windows.h>  // For SetConsoleCtrlHandler
#include <omp.h>
#include <iostream>       // std::cout


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
    int num_threads = 1;
    std::string matrixFile = "s3dkt3m2.mtx";

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--serial-only") {
            runParallel = false;
        }
        else if (arg == "--parallel-only") {
            runSerial = false;
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
#endif


    try {
        // Read the matrix
        std::cout << "Reading matrix file: " << matrixFile << std::endl;
        SparseMatrix sparse = readMatrixMarket(matrixFile);
        sparse.printInfo();

        // Create right-hand side vector b (using all ones)
        int n = sparse.rows;
        std::vector<double> b(n, 1.0);

        // Variables to store timing results
        double serialTime = 0.0;
        double parallelTime = 0.0;
        std::vector<double> x_serial;
        std::vector<double> x_parallel;

        // Run serial implementation
        if (runSerial) {
            std::cout << "\n===== Running Serial Implementation =====\n";
            auto start = std::chrono::high_resolution_clock::now();

            // Convert to CSR format for more efficient operations
            CSRMatrix csrMatrix(sparse, false);

            // Solve using serial BiCGSTAB
            x_serial = biCGSTAB_Serial(csrMatrix, b, 1e-8, 2000, true);

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;
            serialTime = duration.count();
            std::cout << "Serial execution time: " << serialTime << " ms\n";

            // Print solution samples
            std::cout << "Solution vector x (first 10 elements):" << std::endl;
            for (int i = 0; i < min(10, static_cast<int>(x_serial.size())); ++i) {
                std::cout << "x[" << i << "] = " << x_serial[i] << std::endl;
            }

            // Print solution norm for verification
            double norm = 0.0;
            for (const auto& val : x_serial) {
                norm += val * val;
            }
            std::cout << "Serial solution norm: " << std::fixed << std::setprecision(20) << std::sqrt(norm) << std::endl;
        }

        // Run parallel implementation
        if (runParallel) {
            std::cout << "\n===== Running Parallel Implementation =====\n";
            auto start = std::chrono::high_resolution_clock::now();

            // Convert to CSR format with parallel operations
            CSRMatrix csrMatrix(sparse, true);

            // Solve using parallel BiCGSTAB
            x_parallel = biCGSTAB_Parallel(csrMatrix, b, 1e-8, 2000, true);

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;
            parallelTime = duration.count();
            std::cout << "Parallel execution time: " << parallelTime << " ms\n";

            // Print solution samples
            std::cout << "Solution vector x (first 10 elements):" << std::endl;
            for (int i = 0; i < min(10, static_cast<int>(x_parallel.size())); ++i) {
                std::cout << "x[" << i << "] = " << x_parallel[i] << std::endl;
            }

            // Print solution norm for verification
            double norm = 0.0;

#pragma omp parallel for reduction(+:norm)
            for (int i = 0; i < static_cast<int>(x_parallel.size()); i++) {
                norm += x_parallel[i] * x_parallel[i];
            }
            std::cout << "Parallel solution norm: " << std::fixed << std::setprecision(20) << std::sqrt(norm) << std::endl;
        }

        // Compare results
        if (runSerial && runParallel) {
            std::cout << "\n===== Performance Comparison =====\n";
            std::cout << "Serial execution time:   " << serialTime << " ms\n";
            std::cout << "Parallel execution time: " << parallelTime << " ms\n";

            double speedup = serialTime / parallelTime;
            double efficiency = speedup / num_threads;

            std::cout << "Speedup:                 " << speedup << "x\n";
            std::cout << "Parallel efficiency:     " << efficiency * 100 << "%\n";

            // Verify solution correctness
            double diffNorm = 0.0;
            for (size_t i = 0; i < x_serial.size(); i++) {
                double diff = x_serial[i] - x_parallel[i];
                diffNorm += diff * diff;
            }
            diffNorm = std::sqrt(diffNorm);

            std::cout << "Solution difference norm: " << std::fixed << std::setprecision(10) << diffNorm << std::endl;
            if (diffNorm < 1e-6) {
                std::cout << "Serial and parallel solutions match (within tolerance).\n";
            }
            else {
                std::cout << "WARNING: Solutions differ between serial and parallel versions!\n";
            }
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