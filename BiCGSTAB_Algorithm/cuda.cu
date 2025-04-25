#include "common.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// Error checking macro for CUDA calls
#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// CUDA kernel for vector initialization
__global__ void initializeVectorKernel(double* d_vec, int n, double value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_vec[idx] = value;
    }
}

// CUDA kernel for matrix-vector multiplication (CSR format)
__global__ void csrMatVecMulKernel(const int* d_row_ptr, const int* d_col_ind,
    const double* d_values, const double* d_x,
    double* d_y, int rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows) {
        double sum = 0.0;
        int start = d_row_ptr[row];
        int end = d_row_ptr[row + 1];

        for (int j = start; j < end; j++) {
            sum += d_values[j] * d_x[d_col_ind[j]];
        }

        d_y[row] = sum;
    }
}

// CUDA kernel for vector subtraction: y[i] = a[i] - b[i]
__global__ void vectorSubKernel(const double* d_a, const double* d_b, double* d_y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        d_y[idx] = d_a[idx] - d_b[idx];
    }
}

// CUDA kernel for vector addition with scaling: y[i] = alpha*x[i] + beta*y[i]
__global__ void vectorAddScaleKernel(const double* d_x, double* d_y, double alpha,
    double beta, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        d_y[idx] = alpha * d_x[idx] + beta * d_y[idx];
    }
}

// CUDA kernel for vector update: p[i] = r[i] + beta * (p[i] - omega * v[i])
__global__ void updatePKernel(const double* d_r, double* d_p, const double* d_v,
    double beta, double omega, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        d_p[idx] = d_r[idx] + beta * (d_p[idx] - omega * d_v[idx]);
    }
}

// CUDA kernel for vector update: s[i] = r[i] - alpha * v[i]
__global__ void computeSKernel(const double* d_r, const double* d_v, double* d_s,
    double alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        d_s[idx] = d_r[idx] - alpha * d_v[idx];
    }
}

// CUDA kernel for solution update: x[i] += alpha * phat[i] + omega * shat[i]
__global__ void updateSolutionKernel(double* d_x, const double* d_phat, const double* d_shat,
    double alpha, double omega, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        d_x[idx] += alpha * d_phat[idx] + omega * d_shat[idx];
    }
}

// CUDA kernel for residual update: r[i] = s[i] - omega * t[i]
__global__ void updateResidualKernel(double* d_r, const double* d_s, const double* d_t,
    double omega, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        d_r[idx] = d_s[idx] - omega * d_t[idx];
    }
}

// CUDA kernel for applying Jacobi preconditioner: z[i] = r[i] / diag[i]
__global__ void jacobiPreconditionerKernel(const double* d_diag, const double* d_r,
    double* d_z, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        if (fabs(d_diag[idx]) > 1e-10) {
            d_z[idx] = d_r[idx] / d_diag[idx];
        }
        else {
            d_z[idx] = d_r[idx];
        }
    }
}

// CUDA kernel for dot product (first phase - each thread computes partial result)
__global__ void dotProductKernel(const double* d_a, const double* d_b, double* d_result,
    int n) {
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize shared memory
    sdata[tid] = (i < n) ? d_a[i] * d_b[i] : 0.0;
    __syncthreads();

    // Perform reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (tid == 0) {
        d_result[blockIdx.x] = sdata[0];
    }
}

// Compute dot product of two vectors using CUDA
double cudaDotProduct(const double* d_a, const double* d_b, int n,
    double* d_temp, double* h_temp, int blocks, int threads) {
    // First phase: parallel reduction within blocks
    dotProductKernel << <blocks, threads, threads * sizeof(double) >> > (d_a, d_b, d_temp, n);
    CUDA_CHECK(cudaGetLastError());

    // Copy partial results back to host
    CUDA_CHECK(cudaMemcpy(h_temp, d_temp, blocks * sizeof(double), cudaMemcpyDeviceToHost));

    // Finish the reduction on the host
    double result = 0.0;
    for (int i = 0; i < blocks; i++) {
        result += h_temp[i];
    }

    return result;
}

// Compute vector norm using CUDA
double cudaComputeNorm(const double* d_vec, int n, double* d_temp, double* h_temp,
    int blocks, int threads) {
    // Use dot product to compute squared norm
    double norm_squared = cudaDotProduct(d_vec, d_vec, n, d_temp, h_temp, blocks, threads);
    return sqrt(norm_squared);
}

// Copy CSR matrix to GPU
void copyCSRMatrixToDevice(const CSRMatrix& A, int** d_row_ptr, int** d_col_ind,
    double** d_values) {
    // Allocate and copy row pointers
    CUDA_CHECK(cudaMalloc((void**)d_row_ptr, (A.rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(*d_row_ptr, A.row_ptr.data(), (A.rows + 1) * sizeof(int),
        cudaMemcpyHostToDevice));

    // Allocate and copy column indices
    CUDA_CHECK(cudaMalloc((void**)d_col_ind, A.nnz * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(*d_col_ind, A.col_ind.data(), A.nnz * sizeof(int),
        cudaMemcpyHostToDevice));

    // Allocate and copy matrix values
    CUDA_CHECK(cudaMalloc((void**)d_values, A.nnz * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(*d_values, A.values.data(), A.nnz * sizeof(double),
        cudaMemcpyHostToDevice));
}

// Get diagonal elements of matrix
std::vector<double> getCSRDiagonalCUDA(const CSRMatrix& A) {
    std::vector<double> diag(A.rows, 0.0);

    for (int i = 0; i < A.rows; i++) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; j++) {
            if (A.col_ind[j] == i) {
                diag[i] = A.values[j];
                break;
            }
        }
    }

    return diag;
}

// BiCGSTAB (Biconjugate Gradient Stabilized) - CUDA Version
std::vector<double> biCGSTAB_CUDA(const CSRMatrix& A, const std::vector<double>& b,
    double tol, int maxIter, bool usePreconditioner) {

    int n = A.rows;
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;

    // Host memory for vectors
    std::vector<double> x(n, 0.0);  // Initial guess
    std::vector<double> diag = getCSRDiagonalCUDA(A);

    // Device memory for CSR matrix
    int* d_row_ptr, * d_col_ind;
    double* d_values;

    // Device memory for vectors
    double* d_x, * d_b, * d_r, * d_r0, * d_p, * d_v, * d_s, * d_t, * d_phat, * d_shat, * d_diag;

    // Temporary storage for reductions
    double* d_temp, * h_temp;
    int reduction_blocks = (n + threads_per_block - 1) / threads_per_block;
    h_temp = new double[reduction_blocks];

    try {
        // Copy CSR matrix to device
        copyCSRMatrixToDevice(A, &d_row_ptr, &d_col_ind, &d_values);

        // Allocate device memory for vectors
        CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_r, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_r0, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_p, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_v, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_s, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_t, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_phat, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_shat, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_diag, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_temp, reduction_blocks * sizeof(double)));

        // Initialize vectors
        CUDA_CHECK(cudaMemcpy(d_b, b.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_diag, diag.data(), n * sizeof(double), cudaMemcpyHostToDevice));

        // Initialize x with zeros
        initializeVectorKernel << <num_blocks, threads_per_block >> > (d_x, n, 0.0);

        // Initialize p with zeros
        initializeVectorKernel << <num_blocks, threads_per_block >> > (d_p, n, 0.0);

        // Compute r = b - A*x
        csrMatVecMulKernel << <num_blocks, threads_per_block >> > (d_row_ptr, d_col_ind,
            d_values, d_x, d_r, n);
        vectorSubKernel << <num_blocks, threads_per_block >> > (d_b, d_r, d_r, n);

        // Copy r to r0 (shadow residual)
        CUDA_CHECK(cudaMemcpy(d_r0, d_r, n * sizeof(double), cudaMemcpyDeviceToDevice));

        // Compute initial residual norms
        double r0_norm = cudaComputeNorm(d_r, n, d_temp, h_temp, reduction_blocks,
            threads_per_block);
        double b_norm = cudaComputeNorm(d_b, n, d_temp, h_temp, reduction_blocks,
            threads_per_block);

        if (b_norm < 1e-10) {
            b_norm = 1.0;  // Prevent division by zero
        }

        double rel_resid = r0_norm / b_norm;
        if (rel_resid < tol) {
            std::cout << "Initial guess is a good solution, no iterations needed." << std::endl;

            // Copy result back to host and return
            CUDA_CHECK(cudaMemcpy(x.data(), d_x, n * sizeof(double), cudaMemcpyDeviceToHost));

            // Free device memory and return
            // Cleanup code (see below)
            cudaFree(d_row_ptr);
            cudaFree(d_col_ind);
            cudaFree(d_values);
            cudaFree(d_x);
            cudaFree(d_b);
            cudaFree(d_r);
            cudaFree(d_r0);
            cudaFree(d_p);
            cudaFree(d_v);
            cudaFree(d_s);
            cudaFree(d_t);
            cudaFree(d_phat);
            cudaFree(d_shat);
            cudaFree(d_diag);
            cudaFree(d_temp);
            delete[] h_temp;

            return x;  // Already converged
        }

        // Start BiCGSTAB iterations
        double rho_prev = 1.0;
        double alpha = 1.0;
        double omega = 1.0;

        std::cout << "Running BiCGSTAB with CUDA." << std::endl;

        for (int iter = 0; iter < maxIter; iter++) {
            // Compute rho = (r0, r)
            double rho = cudaDotProduct(d_r0, d_r, n, d_temp, h_temp, reduction_blocks,
                threads_per_block);

            if (std::abs(rho) < 1e-10) {
                std::cout << "BiCGSTAB breakdown: rho ~= 0" << std::endl;
                break;
            }

            // Compute beta
            double beta = (rho / rho_prev) * (alpha / omega);
            rho_prev = rho;

            // Update p: p = r + beta * (p - omega * v)
            updatePKernel << <num_blocks, threads_per_block >> > (d_r, d_p, d_v, beta, omega, n);

            // Apply preconditioner to p
            if (usePreconditioner) {
                jacobiPreconditionerKernel << <num_blocks, threads_per_block >> > (d_diag, d_p,
                    d_phat, n);
            }
            else {
                CUDA_CHECK(cudaMemcpy(d_phat, d_p, n * sizeof(double), cudaMemcpyDeviceToDevice));
            }

            // Compute v = A*phat
            csrMatVecMulKernel << <num_blocks, threads_per_block >> > (d_row_ptr, d_col_ind,
                d_values, d_phat, d_v, n);

            // Compute alpha = rho / (r0, v)
            double r0v_dot = cudaDotProduct(d_r0, d_v, n, d_temp, h_temp, reduction_blocks,
                threads_per_block);

            if (std::abs(r0v_dot) < 1e-10) {
                std::cout << "BiCGSTAB breakdown: r0'*v ~= 0" << std::endl;
                break;
            }

            alpha = rho / r0v_dot;

            // Compute s = r - alpha*v
            computeSKernel << <num_blocks, threads_per_block >> > (d_r, d_v, d_s, alpha, n);

            // Check if converged
            double s_norm = cudaComputeNorm(d_s, n, d_temp, h_temp, reduction_blocks,
                threads_per_block);

            if (s_norm / b_norm < tol) {
                // Update solution: x = x + alpha*phat
                vectorAddScaleKernel << <num_blocks, threads_per_block >> > (d_phat, d_x, alpha, 1.0, n);

                std::cout << "Converged after " << iter + 1 << " iterations (early)." << std::endl;
                break;
            }

            // Apply preconditioner to s
            if (usePreconditioner) {
                jacobiPreconditionerKernel << <num_blocks, threads_per_block >> > (d_diag, d_s,
                    d_shat, n);
            }
            else {
                CUDA_CHECK(cudaMemcpy(d_shat, d_s, n * sizeof(double), cudaMemcpyDeviceToDevice));
            }

            // Compute t = A*shat
            csrMatVecMulKernel << <num_blocks, threads_per_block >> > (d_row_ptr, d_col_ind,
                d_values, d_shat, d_t, n);

            // Compute omega = (t, s) / (t, t)
            double ts_dot = cudaDotProduct(d_t, d_s, n, d_temp, h_temp, reduction_blocks,
                threads_per_block);
            double tt_dot = cudaDotProduct(d_t, d_t, n, d_temp, h_temp, reduction_blocks,
                threads_per_block);

            if (std::abs(tt_dot) < 1e-10) {
                std::cout << "BiCGSTAB breakdown: t'*t ~= 0" << std::endl;
                omega = 0.0;
            }
            else {
                omega = ts_dot / tt_dot;
            }

            // Update solution: x = x + alpha*phat + omega*shat
            updateSolutionKernel << <num_blocks, threads_per_block >> > (d_x, d_phat, d_shat,
                alpha, omega, n);

            // Update residual: r = s - omega*t
            updateResidualKernel << <num_blocks, threads_per_block >> > (d_r, d_s, d_t, omega, n);

            // Check convergence
            double r_norm = cudaComputeNorm(d_r, n, d_temp, h_temp, reduction_blocks,
                threads_per_block);
            rel_resid = r_norm / b_norm;

            if ((iter + 1) % 50 == 0) {
                std::cout << "Iteration " << iter + 1 << ": relative residual = "
                    << rel_resid << std::endl;
            }

            if (rel_resid < tol) {
                std::cout << "Converged after " << iter + 1 << " iterations." << std::endl;
                break;
            }

            if (std::abs(omega) < 1e-10) {
                std::cout << "BiCGSTAB breakdown: omega ~= 0" << std::endl;
                break;
            }
        }

        // Check final solution quality
        csrMatVecMulKernel << <num_blocks, threads_per_block >> > (d_row_ptr, d_col_ind,
            d_values, d_x, d_v, n);

        vectorSubKernel << <num_blocks, threads_per_block >> > (d_b, d_v, d_v, n);

        double final_res_norm = cudaComputeNorm(d_v, n, d_temp, h_temp, reduction_blocks,
            threads_per_block) / b_norm;

        std::cout << "Final relative residual: " << final_res_norm << std::endl;

        // Copy result back to host
        CUDA_CHECK(cudaMemcpy(x.data(), d_x, n * sizeof(double), cudaMemcpyDeviceToHost));

    }
    catch (...) {
        std::cerr << "Exception occurred in CUDA BiCGSTAB!" << std::endl;
        throw;
    }

    // Free device memory
    cudaFree(d_row_ptr);
    cudaFree(d_col_ind);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_b);
    cudaFree(d_r);
    cudaFree(d_r0);
    cudaFree(d_p);
    cudaFree(d_v);
    cudaFree(d_s);
    cudaFree(d_t);
    cudaFree(d_phat);
    cudaFree(d_shat);
    cudaFree(d_diag);
    cudaFree(d_temp);
    delete[] h_temp;

    return x;
}