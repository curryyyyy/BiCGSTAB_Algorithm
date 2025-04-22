#include "common.h"
#include <omp.h>

// Parallel implementation of CSRMatrix conversion
void CSRMatrix::convertParallelCSR(const SparseMatrix& mat) {
    rows = mat.rows;
    cols = mat.cols;
    nnz = mat.nnz;

    // Create row pointers
    row_ptr.resize(rows + 1, 0);
    col_ind.resize(nnz);
    values.resize(nnz);

    // Count elements per row - parallelized
#pragma omp parallel for
    for (int i = 0; i < nnz; i++) {
#pragma omp atomic
        row_ptr[mat.row_indices[i] + 1]++;
    }

    // Cumulative sum to get row pointers - must be sequential
    for (int i = 1; i <= rows; i++) {
        row_ptr[i] += row_ptr[i - 1];
    }

    // Need thread-safe insertion into COL and values arrays
    std::vector<int> row_counts(rows, 0);

    // This cannot be safely parallelized without more complex code
    for (int i = 0; i < nnz; i++) {
        int row = mat.row_indices[i];
        int pos = row_ptr[row] + row_counts[row];
        col_ind[pos] = mat.col_indices[i];
        values[pos] = mat.values[i];
        row_counts[row]++;
    }

    // Sort column indices within each row - parallelized
#pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        int start = row_ptr[i];
        int end = row_ptr[i + 1];

        if (end > start) {
            // Create index array for sorting
            std::vector<int> indices(end - start);
            std::iota(indices.begin(), indices.end(), 0);

            // Sort indices based on column values
            std::sort(indices.begin(), indices.end(),
                [&, start](int a, int b) {
                    return col_ind[start + a] < col_ind[start + b];
                });

            // Apply permutation
            std::vector<int> temp_col(end - start);
            std::vector<double> temp_val(end - start);

            for (int j = 0; j < end - start; j++) {
                temp_col[j] = col_ind[start + indices[j]];
                temp_val[j] = values[start + indices[j]];
            }

            // Copy back
            for (int j = 0; j < end - start; j++) {
                col_ind[start + j] = temp_col[j];
                values[start + j] = temp_val[j];
            }
        }
    }
}

// Get diagonal elements - parallel implementation
std::vector<double> CSRMatrix::getDiagonalParallel() const {
    std::vector<double> diag(rows, 0.0);

#pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            if (col_ind[j] == i) {
                diag[i] = values[j];
                break;
            }
        }
    }

    return diag;
}

// Matrix-vector multiplication: y = A*x - Parallel version
void CSRMatrix::multiplyParallel(const std::vector<double>& x, std::vector<double>& y) const {
    std::fill(y.begin(), y.end(), 0.0);

#pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        double sum = 0.0;
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            sum += values[j] * x[col_ind[j]];
        }
        y[i] = sum;
    }
}

// Parallel vector operations
namespace ParallelOps {
    // Compute dot product of two vectors - parallel implementation
    double dotProduct(const std::vector<double>& a, const std::vector<double>& b) {
        double result = 0.0;

#pragma omp parallel for reduction(+:result)
        for (int i = 0; i < static_cast<int>(a.size()); i++) {
            result += a[i] * b[i];
        }
        return result;
    }

    // Compute vector norm - parallel implementation
    double computeNorm(const std::vector<double>& v) {
        return std::sqrt(dotProduct(v, v));
    }

    // Apply Jacobi (diagonal) preconditioner: M^-1 * r - parallel implementation
    void applyJacobiPreconditioner(const std::vector<double>& diag,
        const std::vector<double>& r,
        std::vector<double>& z) {
#pragma omp parallel for
        for (int i = 0; i < static_cast<int>(r.size()); i++) {
            // Avoid division by zero
            if (std::abs(diag[i]) > 1e-10) {
                z[i] = r[i] / diag[i];
            }
            else {
                z[i] = r[i];
            }
        }
    }

    // Apply SSOR preconditioner - only partial parallelization possible
    void applySSORPreconditioner(const CSRMatrix& A,
        const std::vector<double>& r,
        std::vector<double>& z,
        double omega = 1.0) {
        int n = A.rows;
        std::vector<double> y(n, 0.0);
        z.assign(n, 0.0);

        // Forward sweep (Lower triangular solve) - sequential dependency
        for (int i = 0; i < n; i++) {
            double sum = r[i];
            for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; j++) {
                int col = A.col_ind[j];
                if (col < i) {
                    sum -= A.values[j] * z[col];
                }
                else if (col == i) {
                    // Get diagonal element
                    y[i] = A.values[j];
                }
            }
            z[i] = omega * sum / y[i];
        }

        // Backward sweep (Upper triangular solve) - sequential dependency
        for (int i = n - 1; i >= 0; i--) {
            double sum = 0.0;
            for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; j++) {
                int col = A.col_ind[j];
                if (col > i) {
                    sum += A.values[j] * z[col];
                }
            }
            z[i] -= omega * sum / y[i];
        }
    }

    // Vector addition with scaling: y = alpha*x + beta*y - parallel implementation
    void vectorAddScale(const std::vector<double>& x, std::vector<double>& y,
        double alpha = 1.0, double beta = 1.0) {
#pragma omp parallel for
        for (int i = 0; i < static_cast<int>(y.size()); i++) {
            y[i] = alpha * x[i] + beta * y[i];
        }
    }
}

// BiCGSTAB (Biconjugate Gradient Stabilized) - Parallel Version
std::vector<double> biCGSTAB_Parallel(const CSRMatrix& A, const std::vector<double>& b,
    double tol, int maxIter, bool usePreconditioner) {

    using namespace ParallelOps;

    int n = A.rows;
    std::vector<double> x(n, 0.0);  // Initial guess

    // Get diagonal elements for preconditioner
    std::vector<double> diag = A.getDiagonalParallel();

    // Compute r = b - A*x
    std::vector<double> r(n);
    A.multiplyParallel(x, r);

#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        r[i] = b[i] - r[i];
    }

    // Initial residual
    std::vector<double> r0 = r;  // Shadow residual
    double r0_norm = computeNorm(r);
    double b_norm = computeNorm(b);

    if (b_norm < 1e-10) {
        b_norm = 1.0;  // Prevent division by zero
    }

    double rel_resid = r0_norm / b_norm;
    if (rel_resid < tol) {
        std::cout << "Initial guess is a good solution, no iterations needed." << std::endl;
        return x;  // Already converged
    }

    // Create all temporary vectors
    std::vector<double> p(n, 0.0);
    std::vector<double> v(n, 0.0);
    std::vector<double> s(n, 0.0);
    std::vector<double> t(n, 0.0);
    std::vector<double> phat(n, 0.0);
    std::vector<double> shat(n, 0.0);

    // Start BiCGSTAB iterations
    double rho_prev = 1.0;
    double alpha = 1.0;
    double omega = 1.0;

    // Get number of threads
    int num_threads = omp_get_max_threads();
    std::cout << "Running BiCGSTAB with " << num_threads << " threads." << std::endl;

    for (int iter = 0; iter < maxIter; iter++) {
        double rho = dotProduct(r0, r);

        if (std::abs(rho) < 1e-10) {
            std::cout << "BiCGSTAB breakdown: rho ~= 0" << std::endl;
            break;
        }

        // Compute beta
        double beta = (rho / rho_prev) * (alpha / omega);
        rho_prev = rho;

        // Update p: p = r + beta * (p - omega * v)
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            p[i] = r[i] + beta * (p[i] - omega * v[i]);
        }

        // Apply preconditioner to p
        if (usePreconditioner) {
            applyJacobiPreconditioner(diag, p, phat);
        }
        else {
            phat = p;
        }

        // Compute v = A*phat
        A.multiplyParallel(phat, v);

        // Compute alpha
        double r0v_dot = dotProduct(r0, v);
        if (std::abs(r0v_dot) < 1e-10) {
            std::cout << "BiCGSTAB breakdown: r0'*v ~= 0" << std::endl;
            break;
        }

        alpha = rho / r0v_dot;

        // Compute s = r - alpha*v
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            s[i] = r[i] - alpha * v[i];
        }

        // Check if converged
        double s_norm = computeNorm(s);
        if (s_norm / b_norm < tol) {
#pragma omp parallel for
            for (int i = 0; i < n; i++) {
                x[i] += alpha * phat[i];
            }
            std::cout << "Converged after " << iter + 1 << " iterations (early)." << std::endl;
            break;
        }

        // Apply preconditioner to s
        if (usePreconditioner) {
            applyJacobiPreconditioner(diag, s, shat);
        }
        else {
            shat = s;
        }

        // Compute t = A*shat
        A.multiplyParallel(shat, t);

        // Compute omega
        double tt_dot = dotProduct(t, t);
        if (std::abs(tt_dot) < 1e-10) {
            std::cout << "BiCGSTAB breakdown: t'*t ~= 0" << std::endl;
            omega = 0.0;
        }
        else {
            omega = dotProduct(t, s) / tt_dot;
        }

        // Update solution x and residual r in parallel
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            x[i] += alpha * phat[i] + omega * shat[i];
            r[i] = s[i] - omega * t[i];
        }

        // Check convergence
        double r_norm = computeNorm(r);
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
    std::vector<double> final_res(n);
    A.multiplyParallel(x, final_res);

#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        final_res[i] = b[i] - final_res[i];
    }

    double final_res_norm = computeNorm(final_res) / b_norm;
    std::cout << "Final relative residual: " << final_res_norm << std::endl;

    return x;
}