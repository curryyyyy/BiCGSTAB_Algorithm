import os
import subprocess
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from tabulate import tabulate
import time
import re
from concurrent.futures import ProcessPoolExecutor
from matplotlib.ticker import ScalarFormatter

def run_solver(exe_path, matrix_path, output_dir, threads=None, serial_only=False, 
               parallel_only=False, cuda_only=False, hybrid_only=False, no_cuda=False):
    """Run the sparse matrix solver for a single dataset with specified options."""
    
    # Create a unique identifier for this run
    dataset_name = os.path.splitext(os.path.basename(matrix_path))[0]
    run_id = f"{dataset_name}"
    if threads is not None:
        run_id += f"_threads_{threads}"
    
    # Build command-line arguments
    cmd = [exe_path, '--matrix', matrix_path]
    
    if threads is not None:
        cmd.extend(['--threads', str(threads)])
    if serial_only:
        cmd.append('--serial-only')
    if parallel_only:
        cmd.append('--parallel-only')
    if cuda_only:
        cmd.append('--cuda-only')
    if hybrid_only:
        cmd.append('--hybrid-only')
    if no_cuda:
        cmd.append('--no-cuda')
    
    # Create a log file path
    log_file = os.path.join(output_dir, f"{run_id}_log.txt")
    
    # Run the solver
    print(f"Running: {' '.join(cmd)}")
    try:
        with open(log_file, 'w') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
        print(f"Execution completed for {run_id}. Log saved to {log_file}")
        return {
            'dataset': dataset_name,
            'threads': threads,
            'log_file': log_file,
            'success': True,
            'error': None
        }
    except subprocess.CalledProcessError as e:
        print(f"Error executing the program for {run_id}: {e}")
        return {
            'dataset': dataset_name,
            'threads': threads,
            'log_file': log_file,
            'success': False,
            'error': str(e)
        }
    except Exception as e:
        print(f"Unexpected error for {run_id}: {e}")
        return {
            'dataset': dataset_name,
            'threads': threads,
            'log_file': log_file,
            'success': False,
            'error': str(e)
        }

def parse_performance_data(directory):
    """Parse all performance CSV files in the directory."""
    # Find all performance CSV files
    perf_files = glob(os.path.join(directory, "*_performance.csv"))
    
    # Combine all data into a single DataFrame
    all_data = []
    for file in perf_files:
        try:
            df = pd.read_csv(file)
            # Add dataset name from filename
            dataset_name = os.path.basename(file).split('_performance.csv')[0]
            df['dataset'] = dataset_name
            all_data.append(df)
        except Exception as e:
            print(f"Error parsing {file}: {e}")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df
    return None

def extract_convergence_info(log_files):
    """Extract convergence information from log files."""
    convergence_data = []
    
    for log_file in log_files:
        dataset = os.path.basename(log_file).split('_log.txt')[0]
        
        with open(log_file, 'r') as f:
            content = f.read()
            
            # Extract iteration counts for each method
            methods = ['Serial', 'OpenMP', 'Standard CUDA', 'Hybrid CUDA']
            iteration_data = {}
            
            for method in methods:
                pattern = f"Converged after (\\d+) iterations"
                if method == 'Serial':
                    # Look for serial convergence info
                    matches = re.findall(r"===== Running Serial Implementation =====[\s\S]*?Converged after (\d+) iterations", content)
                elif method == 'OpenMP':
                    # Look for OpenMP convergence info
                    matches = re.findall(r"===== Running OpenMP Implementation =====[\s\S]*?Converged after (\d+) iterations", content)
                elif method == 'Standard CUDA':
                    # Look for CUDA convergence info
                    matches = re.findall(r"===== Running Standard CUDA Implementation =====[\s\S]*?Converged after (\d+) iterations", content)
                elif method == 'Hybrid CUDA':
                    # Look for Hybrid CUDA convergence info
                    matches = re.findall(r"===== Running Hybrid CUDA Implementation =====[\s\S]*?Converged after (\d+) iterations", content)
                
                if matches:
                    iteration_data[method] = int(matches[0])
                else:
                    iteration_data[method] = None
            
            # Extract final residuals
            residual_data = {}
            for method in methods:
                pattern = f"{method} solution norm: ([0-9.e+-]+)"
                matches = re.findall(pattern, content)
                if matches:
                    residual_data[method] = float(matches[0])
                else:
                    residual_data[method] = None
                    
            # Add convergence data
            convergence_data.append({
                'dataset': dataset,
                'iterations': iteration_data,
                'residuals': residual_data
            })
    
    return convergence_data

def generate_plots(performance_data, output_dir):
    """Generate performance comparison plots."""
    if performance_data is None or performance_data.empty:
        print("No performance data available for plotting")
        return
    
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    
    # Plot 1: Solver time comparison across methods for each dataset
    datasets = performance_data['dataset'].unique()
    
    # Prepare a DataFrame for plotting
    for dataset in datasets:
        dataset_data = performance_data[performance_data['dataset'] == dataset]
        
        # Create bar chart comparing methods
        plt.figure(figsize=(12, 8))
        methods = dataset_data['method'].unique()
        solver_times = [dataset_data[dataset_data['method'] == method]['solver_time_ms'].values[0] 
                        for method in methods]
        
        # Calculate speedups relative to serial (if present)
        serial_time = None
        if 'serial' in methods:
            serial_time = dataset_data[dataset_data['method'] == 'serial']['solver_time_ms'].values[0]
            
        plt.bar(methods, solver_times, color=['blue', 'green', 'red', 'purple'])
        plt.title(f'Solver Time Comparison - {dataset}')
        plt.xlabel('Implementation Method')
        plt.ylabel('Solver Time (ms)')
        plt.yscale('log')
        plt.grid(True, which="both", ls="--", alpha=0.3)
        
        # Add exact times on top of bars
        for i, time_value in enumerate(solver_times):
            plt.text(i, time_value*1.05, f'{time_value:.2f} ms', 
                    ha='center', va='bottom', rotation=0, fontweight='bold')
            
            # Add speedup if serial data is available
            if serial_time is not None and methods[i] != 'serial':
                speedup = serial_time / time_value
                plt.text(i, time_value*0.5, f'Speedup: {speedup:.2f}x', 
                        ha='center', va='center', color='white', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "plots", f"{dataset}_method_comparison.png"))
        plt.close()
    
    # If we have OpenMP data with different thread counts, create scaling plots
    openmp_data = performance_data[performance_data['method'] == 'openmp']
    if not openmp_data.empty and 'threads' in openmp_data.columns:
        for dataset in openmp_data['dataset'].unique():
            dataset_openmp = openmp_data[openmp_data['dataset'] == dataset]
            if len(dataset_openmp['threads'].unique()) > 1:
                plt.figure(figsize=(10, 6))
                
                thread_counts = sorted(dataset_openmp['threads'].unique())
                times = [dataset_openmp[dataset_openmp['threads'] == t]['solver_time_ms'].values[0] 
                         for t in thread_counts]
                
                # Calculate speedups
                base_time = times[0]  # Time with minimum threads
                speedups = [base_time / t for t in times]
                
                # Create two subplots
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                
                # Time vs threads
                ax1.plot(thread_counts, times, 'o-', color='blue', linewidth=2)
                ax1.set_title(f'OpenMP Performance Scaling - {dataset}')
                ax1.set_xlabel('Thread Count')
                ax1.set_ylabel('Solver Time (ms)')
                ax1.set_xscale('log', base=2)
                ax1.set_yscale('log')
                ax1.grid(True, which="both", ls="--", alpha=0.3)
                
                # Speedup vs threads - with ideal scaling line
                ax2.plot(thread_counts, speedups, 'o-', color='green', linewidth=2, label='Actual')
                ax2.plot(thread_counts, thread_counts, '--', color='red', label='Ideal')
                ax2.set_title(f'OpenMP Speedup - {dataset}')
                ax2.set_xlabel('Thread Count')
                ax2.set_ylabel('Speedup')
                ax2.set_xscale('log', base=2)
                ax2.set_yscale('log', base=2)
                ax2.grid(True, which="both", ls="--", alpha=0.3)
                ax2.legend()
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "plots", f"{dataset}_openmp_scaling.png"))
                plt.close()

def generate_summary_report(performance_data, convergence_data, output_dir):
    """Generate a summary report of all runs."""
    if performance_data is None or performance_data.empty:
        print("No performance data available for summary report")
        return
    
    summary_file = os.path.join(output_dir, "summary_report.md")
    
    with open(summary_file, 'w') as f:
        # Write header
        f.write("# Sparse Matrix Solver Performance Summary\n\n")
        f.write(f"Report generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overall summary
        f.write("## Overall Summary\n\n")
        
        datasets = performance_data['dataset'].unique()
        f.write(f"- Total datasets processed: {len(datasets)}\n")
        f.write(f"- Methods evaluated: {', '.join(performance_data['method'].unique())}\n\n")
        
        # Performance table by dataset
        f.write("## Performance Comparison by Dataset\n\n")
        
        summary_rows = []
        for dataset in datasets:
            dataset_data = performance_data[performance_data['dataset'] == dataset]
            
            # Get matrix size
            matrix_size = dataset_data['matrix_size'].iloc[0]
            
            # Get solver times for each method (if available)
            methods = ['serial', 'openmp', 'standard_cuda', 'hybrid_cuda']
            times = []
            
            for method in methods:
                method_data = dataset_data[dataset_data['method'] == method]
                if not method_data.empty:
                    times.append(f"{method_data['solver_time_ms'].values[0]:.2f}")
                else:
                    times.append("N/A")
            
            summary_rows.append([dataset, matrix_size] + times)
        
        # Write the table
        headers = ["Dataset", "Matrix Size", "Serial (ms)", "OpenMP (ms)", "CUDA (ms)", "Hybrid (ms)"]
        f.write(tabulate(summary_rows, headers=headers, tablefmt="pipe"))
        f.write("\n\n")
        
        # Add speedup analysis
        f.write("## Speedup Analysis\n\n")
        
        speedup_rows = []
        for dataset in datasets:
            dataset_data = performance_data[performance_data['dataset'] == dataset]
            
            # Check if serial time is available (as the baseline)
            serial_data = dataset_data[dataset_data['method'] == 'serial']
            if serial_data.empty:
                continue
                
            serial_time = serial_data['solver_time_ms'].values[0]
            
            # Calculate speedups
            speedups = []
            for method in ['openmp', 'standard_cuda', 'hybrid_cuda']:
                method_data = dataset_data[dataset_data['method'] == method]
                if not method_data.empty:
                    method_time = method_data['solver_time_ms'].values[0]
                    speedup = serial_time / method_time
                    speedups.append(f"{speedup:.2f}x")
                else:
                    speedups.append("N/A")
            
            speedup_rows.append([dataset] + speedups)
        
        # Write the speedup table
        headers = ["Dataset", "OpenMP Speedup", "CUDA Speedup", "Hybrid Speedup"]
        f.write(tabulate(speedup_rows, headers=headers, tablefmt="pipe"))
        f.write("\n\n")
        
        # Add convergence information
        if convergence_data:
            f.write("## Convergence Information\n\n")
            
            conv_rows = []
            for data in convergence_data:
                dataset = data['dataset']
                iterations = data['iterations']
                
                iter_values = []
                for method in ['Serial', 'OpenMP', 'Standard CUDA', 'Hybrid CUDA']:
                    if method in iterations and iterations[method] is not None:
                        iter_values.append(str(iterations[method]))
                    else:
                        iter_values.append("N/A")
                
                conv_rows.append([dataset] + iter_values)
            
            # Write the convergence table
            headers = ["Dataset", "Serial Iterations", "OpenMP Iterations", "CUDA Iterations", "Hybrid Iterations"]
            f.write(tabulate(conv_rows, headers=headers, tablefmt="pipe"))
            f.write("\n\n")
        
        # Conclusion and recommendations
        f.write("## Conclusion\n\n")
        
        # Find best performing method overall
        best_methods = {}
        for dataset in datasets:
            dataset_data = performance_data[performance_data['dataset'] == dataset]
            if not dataset_data.empty:
                best_method = dataset_data.loc[dataset_data['solver_time_ms'].idxmin()]['method']
                if best_method in best_methods:
                    best_methods[best_method] += 1
                else:
                    best_methods[best_method] = 1
        
        if best_methods:
            overall_best = max(best_methods.items(), key=lambda x: x[1])[0]
            f.write(f"- Overall best performing method: **{overall_best}**\n")
            f.write(f"- Method performance distribution: {best_methods}\n\n")
        
        f.write("### Recommendations\n\n")
        f.write("Based on the performance analysis:\n\n")
        
        if 'hybrid_cuda' in best_methods and best_methods.get('hybrid_cuda', 0) > best_methods.get('standard_cuda', 0):
            f.write("- The hybrid CUDA implementation (OpenMP + CUDA) generally outperforms the standard CUDA implementation.\n")
        
        if 'openmp' in best_methods:
            f.write("- For systems without GPUs, the OpenMP implementation provides good performance.\n")
        
        f.write("\n*This report was automatically generated.*\n")
    
    print(f"Summary report generated: {summary_file}")
    return summary_file

def main():
    parser = argparse.ArgumentParser(description='Run sparse matrix solver with multiple datasets')
    parser.add_argument('--exe', type=str, required=True, help='Path to the executable')
    parser.add_argument('--datasets-dir', type=str, required=True, help='Directory containing .mtx datasets')
    parser.add_argument('--output-dir', type=str, default='results', help='Directory to store results')
    parser.add_argument('--threads', type=str, default=None, 
                        help='Number of threads for OpenMP (comma-separated for multiple counts, e.g., "1,2,4,8")')
    parser.add_argument('--serial-only', action='store_true', help='Run only serial version')
    parser.add_argument('--parallel-only', action='store_true', help='Run only parallel version')
    parser.add_argument('--cuda-only', action='store_true', help='Run only CUDA version')
    parser.add_argument('--hybrid-only', action='store_true', help='Run only hybrid version')
    parser.add_argument('--no-cuda', action='store_true', help='Don\'t run CUDA versions')
    parser.add_argument('--max-workers', type=int, default=1, 
                        help='Maximum number of concurrent processes to run (default: 1)')
    parser.add_argument('--auto-threads', action='store_true',
                        help='Automatically test with thread counts 1, 2, 4, ..., up to the number of CPU cores')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all .mtx files in the datasets directory
    dataset_files = glob(os.path.join(args.datasets_dir, '*.mtx'))
    
    if not dataset_files:
        print(f"No .mtx files found in {args.datasets_dir}")
        return
    
    print(f"Found {len(dataset_files)} dataset files to process")
    
    # Determine thread counts to test
    thread_counts = []
    if args.threads:
        thread_counts = [int(t) for t in args.threads.split(',')]
    elif args.auto_threads:
        import multiprocessing
        max_cores = multiprocessing.cpu_count()
        # Generate power-of-2 thread counts: 1, 2, 4, 8, ...
        t = 1
        while t <= max_cores:
            thread_counts.append(t)
            t *= 2
    else:
        thread_counts = [None]  # Use default thread count
    
    # Prepare run configurations
    run_configs = []
    for dataset_file in dataset_files:
        for thread_count in thread_counts:
            run_configs.append({
                'exe_path': args.exe,
                'matrix_path': dataset_file,
                'output_dir': args.output_dir,
                'threads': thread_count,
                'serial_only': args.serial_only,
                'parallel_only': args.parallel_only,
                'cuda_only': args.cuda_only,
                'hybrid_only': args.hybrid_only,
                'no_cuda': args.no_cuda
            })
    
    # Execute runs in parallel or sequentially
    results = []
    if args.max_workers > 1:
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = []
            for config in run_configs:
                futures.append(executor.submit(run_solver, **config))
            
            # Collect results as they complete
            for future in futures:
                results.append(future.result())
    else:
        for config in run_configs:
            results.append(run_solver(**config))
    
    # Process the results
    successful_runs = [r for r in results if r['success']]
    failed_runs = [r for r in results if not r['success']]
    
    print(f"\nRun summary:")
    print(f"- Successful runs: {len(successful_runs)}")
    print(f"- Failed runs: {len(failed_runs)}")
    
    if failed_runs:
        print("\nFailed runs:")
        for run in failed_runs:
            print(f"- Dataset: {run['dataset']}, Threads: {run['threads']}, Error: {run['error']}")
    
    # Parse performance data
    print("\nParsing performance data...")
    performance_data = parse_performance_data(args.output_dir)
    
    # Extract convergence information from log files
    log_files = [r['log_file'] for r in successful_runs]
    convergence_data = extract_convergence_info(log_files)
    
    # Generate plots
    print("Generating performance plots...")
    generate_plots(performance_data, args.output_dir)
    
    # Generate summary report
    print("Generating summary report...")
    summary_file = generate_summary_report(performance_data, convergence_data, args.output_dir)
    
    print(f"\nAnalysis complete! Check {args.output_dir} for results.")
    print(f"Summary report: {summary_file}")
    print(f"Performance plots: {os.path.join(args.output_dir, 'plots')}")

if __name__ == "__main__":
    main()