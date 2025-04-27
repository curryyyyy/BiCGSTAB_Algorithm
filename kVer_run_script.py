import subprocess
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import glob
from pathlib import Path

def run_solver(executable_path, matrix_file, implementation=None, threads=None):
    """Run the BiCGSTAB solver with specified parameters"""
    cmd = [executable_path]
    
    # Add implementation flag
    if implementation:
        if implementation == "serial":
            cmd.append("--serial-only")
        elif implementation == "parallel":
            cmd.append("--parallel-only")
        elif implementation == "cuda":
            cmd.append("--cuda-only")
        elif implementation == "hybrid":
            cmd.append("--hybrid-only")
    
    # Add matrix file path
    if matrix_file:
        cmd.extend(["--matrix", matrix_file])
    
    # Add thread count if specified
    if threads is not None:
        cmd.extend(["--threads", str(threads)])
    
    print(f"\n=== Running BiCGSTAB solver with parameters ===")
    print(f"Matrix file: {matrix_file}")
    if implementation:
        print(f"Implementation: {implementation}")
    if threads:
        print(f"Threads: {threads}")
    print("=" * 40)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:")
            print(result.stderr)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running the executable: {e}")
        if e.stdout:
            print("Standard output:")
            print(e.stdout)
        if e.stderr:
            print("Error output:")
            print(e.stderr)
        return False, None

def extract_performance_data(output_text, matrix_name, implementation, threads=None):
    """Extract performance data from program output"""
    data = {
        'matrix': matrix_name,
        'implementation': implementation,
        'threads': threads if threads is not None else 'N/A',
        'file_read_time_ms': None,
        'csr_conversion_time_ms': None,
        'solver_time_ms': None, 
        'total_time_ms': None
    }
    
    # Extract performance metrics using regex
    file_read_match = re.search(r'(Serial|Parallel|CUDA|Hybrid) file reading time: ([\d.]+) ms', output_text)
    if file_read_match:
        data['file_read_time_ms'] = float(file_read_match.group(2))
    
    csr_match = re.search(r'(Serial|Parallel|CUDA|Hybrid|OpenMP|Standard CUDA) CSR conversion time: ([\d.]+) ms', output_text)
    if csr_match:
        data['csr_conversion_time_ms'] = float(csr_match.group(2))
    
    solver_match = re.search(r'(Serial|Parallel|CUDA|Hybrid|OpenMP|Standard CUDA) solver time: ([\d.]+) ms', output_text)
    if solver_match:
        data['solver_time_ms'] = float(solver_match.group(2))
    
    total_match = re.search(r'(Serial|Parallel|CUDA|Hybrid|OpenMP|Standard CUDA) total time: ([\d.]+) ms', output_text)
    if total_match:
        data['total_time_ms'] = float(total_match.group(2))
    
    # Extract matrix size
    matrix_size_match = re.search(r'Matrix dimensions: (\d+) x (\d+)', output_text)
    if matrix_size_match:
        rows = int(matrix_size_match.group(1))
        cols = int(matrix_size_match.group(2))
        data['matrix_size'] = rows
        data['matrix_dimensions'] = f"{rows}x{cols}"
    
    # Extract convergence info
    converged_match = re.search(r'Converged after (\d+) iterations', output_text)
    if converged_match:
        data['iterations'] = int(converged_match.group(1))
    
    # Extract residual info
    residual_match = re.search(r'Final relative residual: ([\d.e+-]+)', output_text)
    if residual_match:
        data['final_residual'] = float(residual_match.group(1))
    
    return data

def combine_results(results_list):
    """Combine all results into a single DataFrame"""
    df = pd.DataFrame(results_list)
    
    # Calculate speedups relative to serial
    implementations = df['implementation'].unique()
    if 'serial' in implementations:
        # Group by matrix
        for matrix in df['matrix'].unique():
            serial_time = df[(df['matrix'] == matrix) & (df['implementation'] == 'serial')]['solver_time_ms'].values
            if len(serial_time) > 0:
                serial_time = serial_time[0]
                for impl in implementations:
                    if impl != 'serial':
                        mask = (df['matrix'] == matrix) & (df['implementation'] == impl)
                        if any(mask):
                            df.loc[mask, f'{impl}_speedup'] = serial_time / df.loc[mask, 'solver_time_ms']
    
    return df

def create_visualizations(df, output_dir):
    """Create visualizations based on the performance results"""
    # Create output directory for graphs
    graphs_dir = os.path.join(output_dir, "graphs")
    os.makedirs(graphs_dir, exist_ok=True)
    
    # Check which implementations were run
    implementations = df['implementation'].unique()
    
    # Get matrix names and sizes for sorting
    matrices = df['matrix'].unique()
    
    # Create a lookup for matrix sizes to use for sorting
    size_lookup = {}
    for matrix in matrices:
        size = df[df['matrix'] == matrix]['matrix_size'].iloc[0]
        size_lookup[matrix] = size
    
    # Sort matrices by size
    sorted_matrices = sorted(matrices, key=lambda x: size_lookup[x])
    
    # =========================================================
    # 1. Bar chart comparing solver times across implementations
    # =========================================================
    plt.figure(figsize=(14, 8))
    bar_width = 0.8 / len(implementations)
    
    for i, impl in enumerate(implementations):
        impl_data = df[df['implementation'] == impl]
        x = np.arange(len(matrices))
        solver_times = [impl_data[impl_data['matrix'] == m]['solver_time_ms'].mean() / 1000 for m in matrices]
        
        plt.bar(x + (i - len(implementations)/2 + 0.5) * bar_width, 
                solver_times, 
                width=bar_width, 
                label=impl.capitalize())
    
    plt.xlabel('Matrix')
    plt.ylabel('Solver Time (seconds)')
    plt.title('BiCGSTAB Solver Time by Implementation')
    plt.xticks(np.arange(len(matrices)), matrices, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{graphs_dir}/solver_times_by_implementation.png")
    plt.close()
    
    # =========================================================
    # 2. Line graph comparing solver times by matrix size
    # =========================================================
    plt.figure(figsize=(12, 7))
    
    for impl in implementations:
        impl_data = df[df['implementation'] == impl]
        # Sort by matrix size
        impl_data = impl_data.sort_values(by='matrix_size')
        # Plot line
        plt.plot(impl_data['matrix_size'], 
                impl_data['solver_time_ms'] / 1000, 
                marker='o', 
                linestyle='-', 
                label=impl.capitalize())
    
    plt.xlabel('Matrix Size (Number of Rows)')
    plt.ylabel('Solver Time (seconds)')
    plt.title('BiCGSTAB Solver Performance vs Matrix Size')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{graphs_dir}/solver_time_vs_matrix_size.png")
    plt.close()
    
    # =========================================================
    # 3. Bar chart of speedups compared to serial implementation
    # =========================================================
    if 'serial' in implementations and len(implementations) > 1:
        plt.figure(figsize=(14, 8))
        bar_width = 0.8 / (len(implementations) - 1)
        x = np.arange(len(matrices))
        
        bar_positions = []
        bar_heights = []
        bar_labels = []
        bar_colors = ['blue', 'green', 'red', 'purple']
        
        i = 0
        for impl in implementations:
            if impl != 'serial':
                speedup_col = f'{impl}_speedup'
                if speedup_col in df.columns:
                    for m_idx, matrix in enumerate(matrices):
                        matrix_data = df[(df['matrix'] == matrix) & (df['implementation'] == impl)]
                        if not matrix_data.empty and speedup_col in matrix_data.columns:
                            speedup = matrix_data[speedup_col].values[0]
                            bar_positions.append(m_idx + (i - (len(implementations)-1)/2 + 0.5) * bar_width)
                            bar_heights.append(speedup)
                            bar_labels.append(impl.capitalize())
                    i += 1
        
        plt.bar(bar_positions, bar_heights, width=bar_width, color=[bar_colors[implementations.tolist().index(label.lower())-1] for label in bar_labels])
        
        # Create legend with unique entries
        unique_labels = list(set(bar_labels))
        plt.legend(handles=[plt.Rectangle((0,0), 1, 1, color=bar_colors[implementations.tolist().index(label.lower())-1]) for label in unique_labels], 
                   labels=unique_labels)
        
        plt.xlabel('Matrix')
        plt.ylabel('Speedup Factor (compared to Serial)')
        plt.title('Implementation Speedup Factors')
        plt.xticks(np.arange(len(matrices)), matrices, rotation=45, ha='right')
        plt.grid(True, axis='y', alpha=0.3)
        plt.axhline(y=1, color='k', linestyle='--', alpha=0.5)  # Baseline for no speedup
        plt.tight_layout()
        plt.savefig(f"{graphs_dir}/speedup_factors.png")
        plt.close()
    
    # =========================================================
    # 4. Stacked bar chart showing time breakdown
    # =========================================================
    plt.figure(figsize=(14, 8))
    
    for i, impl in enumerate(implementations):
        impl_data = df[df['implementation'] == impl]
        x = np.arange(len(matrices))
        
        # Get timing components, filling with 0 if missing
        file_read_times = [impl_data[impl_data['matrix'] == m]['file_read_time_ms'].iloc[0]/1000 if not impl_data[impl_data['matrix'] == m].empty and not impl_data[impl_data['matrix'] == m]['file_read_time_ms'].isna().iloc[0] else 0 for m in matrices]
        csr_times = [impl_data[impl_data['matrix'] == m]['csr_conversion_time_ms'].iloc[0]/1000 if not impl_data[impl_data['matrix'] == m].empty and not impl_data[impl_data['matrix'] == m]['csr_conversion_time_ms'].isna().iloc[0] else 0 for m in matrices]
        solver_times = [impl_data[impl_data['matrix'] == m]['solver_time_ms'].iloc[0]/1000 if not impl_data[impl_data['matrix'] == m].empty and not impl_data[impl_data['matrix'] == m]['solver_time_ms'].isna().iloc[0] else 0 for m in matrices]
        
        # Create position for this implementation's set of bars
        pos = x + (i - len(implementations)/2 + 0.5) * 0.8/len(implementations)
        
        # Plot stacked bars
        plt.bar(pos, file_read_times, width=0.8/len(implementations), label=f'{impl.capitalize()} - File Read' if i == 0 else "", color='lightblue')
        plt.bar(pos, csr_times, width=0.8/len(implementations), bottom=file_read_times, label=f'{impl.capitalize()} - CSR Conversion' if i == 0 else "", color='lightgreen')
        plt.bar(pos, solver_times, width=0.8/len(implementations), bottom=[a+b for a,b in zip(file_read_times, csr_times)], label=f'{impl.capitalize()} - Solver' if i == 0 else "", color='coral')
    
    plt.xlabel('Matrix')
    plt.ylabel('Time (seconds)')
    plt.title('BiCGSTAB Performance Breakdown')
    plt.xticks(np.arange(len(matrices)), matrices, rotation=45, ha='right')
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightblue', label='File Read'),
        Patch(facecolor='lightgreen', label='CSR Conversion'),
        Patch(facecolor='coral', label='Solver')
    ]
    
    # Add implementation labels
    for i, impl in enumerate(implementations):
        legend_elements.append(plt.Line2D([0], [0], color='black', lw=4, label=impl.capitalize()))
    
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{graphs_dir}/performance_breakdown.png")
    plt.close()
    
    # =========================================================
    # 5. Thread scaling for OpenMP (if multiple thread counts were tested)
    # =========================================================
    if 'parallel' in implementations:
        parallel_data = df[df['implementation'] == 'parallel']
        thread_counts = parallel_data['threads'].unique()
        
        if len(thread_counts) > 1 and all(tc != 'N/A' for tc in thread_counts):
            thread_counts = sorted([int(tc) for tc in thread_counts if tc != 'N/A'])
            
            plt.figure(figsize=(10, 6))
            
            # For each matrix, plot performance vs thread count
            for matrix in matrices:
                matrix_data = parallel_data[parallel_data['matrix'] == matrix].copy()
                if len(matrix_data) > 1:  # Only if we have multiple thread counts for this matrix
                    matrix_data['threads'] = matrix_data['threads'].astype(int)
                    matrix_data = matrix_data.sort_values(by='threads')
                    
                    # Calculate speedup relative to single thread
                    if 1 in matrix_data['threads'].values:
                        single_thread_time = matrix_data[matrix_data['threads'] == 1]['solver_time_ms'].values[0]
                        plt.plot(matrix_data['threads'], 
                                single_thread_time / matrix_data['solver_time_ms'], 
                                marker='o', linestyle='-', label=matrix)
            
            # Add ideal scaling line
            max_threads = max(thread_counts)
            plt.plot([1, max_threads], [1, max_threads], 'k--', alpha=0.5, label='Ideal Scaling')
            
            plt.xlabel('Number of Threads')
            plt.ylabel('Speedup Factor')
            plt.title('OpenMP Thread Scaling')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{graphs_dir}/openmp_thread_scaling.png")
            plt.close()
    
    # =========================================================
    # 6. Convergence behavior - iterations vs matrix size
    # =========================================================
    if 'iterations' in df.columns:
        plt.figure(figsize=(12, 6))
        
        for impl in implementations:
            impl_data = df[df['implementation'] == impl]
            if 'iterations' in impl_data.columns:
                # Sort by matrix size
                impl_data = impl_data.sort_values(by='matrix_size')
                plt.plot(impl_data['matrix_size'], 
                        impl_data['iterations'], 
                        marker='o', 
                        linestyle='-', 
                        label=impl.capitalize())
        
        plt.xlabel('Matrix Size (Number of Rows)')
        plt.ylabel('Iterations to Convergence')
        plt.title('BiCGSTAB Convergence Behavior')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{graphs_dir}/convergence_iterations.png")
        plt.close()
    
    print(f"Graphs generated successfully in directory: {graphs_dir}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Run BiCGSTAB solver with multiple parameter combinations and generate performance visualizations")
    parser.add_argument("--executable", "-e", required=True, help="Path to the BiCGSTAB executable")
    parser.add_argument("--matrix-dir", "-d", help="Directory containing matrix files")
    parser.add_argument("--matrix-file", "-m", help="Single matrix file to test")
    parser.add_argument("--output", "-o", help="Output directory for results", default="./results")
    parser.add_argument("--implementations", "-i", nargs="+", default=["serial", "parallel", "cuda", "hybrid"], 
                      help="List of implementations to test (serial, parallel, cuda, hybrid, all)")
    parser.add_argument("--threads", "-t", nargs="+", type=int, default=[4],
                      help="List of thread counts to test for parallel implementation")
    parser.add_argument("--visualize-only", action="store_true", 
                      help="Skip running solver and only generate visualization from existing CSV")
    parser.add_argument("--results-file", "-r", help="CSV file with existing results (for visualize-only mode)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Path for results CSV
    results_csv = args.results_file if args.results_file else os.path.join(args.output, "bicgstab_results.csv")
    
    if not args.visualize_only:
        # Verify executable exists
        if not os.path.isfile(args.executable):
            print(f"Error: Executable {args.executable} not found")
            return
        
        # Get list of matrix files to process
        matrix_files = []
        if args.matrix_file:
            if os.path.isfile(args.matrix_file):
                matrix_files.append(args.matrix_file)
            else:
                print(f"Error: Matrix file {args.matrix_file} not found")
                return
        elif args.matrix_dir:
            if os.path.isdir(args.matrix_dir):
                matrix_files = glob.glob(os.path.join(args.matrix_dir, "*.mtx"))
                if not matrix_files:
                    print(f"No .mtx files found in directory {args.matrix_dir}")
                    return
            else:
                print(f"Error: Matrix directory {args.matrix_dir} not found")
                return
        else:
            print("Error: Either --matrix-file or --matrix-dir must be specified")
            return
        
        # Store all results
        all_results = []
        
        # Track total number of runs
        implementations = [impl for impl in args.implementations if impl != "all"]
        if "all" in args.implementations:
            implementations = ["serial", "parallel", "cuda", "hybrid"]
        
        # Calculate total runs
        total_runs = len(matrix_files) * len(implementations)
        if "parallel" in implementations:
            total_runs += len(matrix_files) * (len(args.threads) - 1)  # -1 because one thread count is default
        
        current_run = 1
        
        # Process each matrix file
        for matrix_file in matrix_files:
            matrix_name = os.path.basename(matrix_file)
            
            # Run with each implementation
            for implementation in implementations:
                print(f"\nRun {current_run}/{total_runs}")
                current_run += 1
                
                # For parallel implementation, test different thread counts
                if implementation == "parallel" and args.threads:
                    for thread_count in args.threads:
                        print(f"Testing parallel implementation with {thread_count} threads")
                        success, output = run_solver(args.executable, matrix_file, implementation, thread_count)
                        
                        if success and output:
                            # Extract and store results
                            result_data = extract_performance_data(output, matrix_name, implementation, thread_count)
                            all_results.append(result_data)
                else:
                    # Run with default settings
                    success, output = run_solver(args.executable, matrix_file, implementation)
                    
                    if success and output:
                        # Extract and store results
                        result_data = extract_performance_data(output, matrix_name, implementation)
                        all_results.append(result_data)
        
        # Combine all results and save to CSV
        if all_results:
            results_df = combine_results(all_results)
            results_df.to_csv(results_csv, index=False)
            print(f"\nResults saved to {results_csv}")
        else:
            print("No results collected. Check for errors in the execution.")
            return
    
    # Generate visualizations
    if args.visualize_only:
        if not os.path.exists(results_csv):
            print(f"Error: Results CSV file {results_csv} not found")
            return
        
        results_df = pd.read_csv(results_csv)
    else:
        # Use the dataframe we already created
        results_df = results_df if 'results_df' in locals() else pd.read_csv(results_csv)
    
    print("\nGenerating visualizations...")
    if create_visualizations(results_df, args.output):
        print("Visualization process completed successfully!")
    else:
        print("Failed to generate visualizations.")

if __name__ == "__main__":
    main()