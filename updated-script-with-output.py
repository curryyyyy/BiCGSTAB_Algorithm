import os
import subprocess
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import time
import re
import sys
from concurrent.futures import ProcessPoolExecutor

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
    print(f"\n[{time.strftime('%H:%M:%S')}] STARTING: {os.path.basename(matrix_path)} (threads: {threads if threads else 'default'})")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 80)  # Divider line for better readability
    
    start_time = time.time()
    try:
        # Open the log file for writing
        log_file_handle = open(log_file, 'w')
        
        # Run the process with real-time output display
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                  text=True, bufsize=1, universal_newlines=True)
        
        # Process and display output in real-time
        for line in process.stdout:
            # Write to log file
            log_file_handle.write(line)
            log_file_handle.flush()
            
            # Display in console (without extra newlines)
            sys.stdout.write(line)
            sys.stdout.flush()
        
        # Wait for the process to complete
        process.wait()
        
        # Close log file
        log_file_handle.close()
        
        # Process completed
        elapsed = time.time() - start_time
        print(f"\n[{time.strftime('%H:%M:%S')}] COMPLETED: {os.path.basename(matrix_path)} in {elapsed:.1f} seconds")
        
        # Check return code
        if process.returncode != 0:
            print(f"Warning: Process exited with code {process.returncode}")
                
        print(f"Log saved to {log_file}")
        return {
            'dataset': dataset_name,
            'threads': threads,
            'log_file': log_file,
            'success': True,
            'error': None,
            'runtime': elapsed
        }
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n[{time.strftime('%H:%M:%S')}] ERROR: Process failed after {elapsed:.1f} seconds")
        print(f"Error executing the program for {run_id}: {e}")
        return {
            'dataset': dataset_name,
            'threads': threads,
            'log_file': log_file,
            'success': False,
            'error': str(e),
            'runtime': elapsed
        }
    except KeyboardInterrupt:
        print(f"\n[{time.strftime('%H:%M:%S')}] INTERRUPTED: Process was manually stopped")
        raise
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n[{time.strftime('%H:%M:%S')}] ERROR: Unexpected error after {elapsed:.1f} seconds")
        print(f"Unexpected error for {run_id}: {e}")
        return {
            'dataset': dataset_name,
            'threads': threads,
            'log_file': log_file,
            'success': False,
            'error': str(e),
            'runtime': elapsed
        }

def extract_performance_data_from_logs(log_files, output_dir):
    """Extract performance data directly from log files and save to CSV."""
    all_data = []
    
    for log_file in log_files:
        dataset_name = os.path.basename(log_file).split('_log')[0]
        
        print(f"Extracting performance data from {os.path.basename(log_file)}")
        
        with open(log_file, 'r') as f:
            content = f.read()
            
            # Extract matrix dimensions
            matrix_dims_match = re.search(r'Matrix dimensions: (\d+) x (\d+)', content)
            matrix_size = 0
            if matrix_dims_match:
                rows = int(matrix_dims_match.group(1))
                matrix_size = rows
            
            # Try to extract size from dataset name if not found in log
            if matrix_size == 0:
                size_match = re.search(r'(\d+)mb', dataset_name.lower())
                if size_match:
                    matrix_size = int(size_match.group(1))
            
            # Extract performance data for each implementation
            for method, method_name in [
                ('Serial', 'serial'),
                ('OpenMP', 'openmp'), 
                ('Standard CUDA', 'standard_cuda'),
                ('Hybrid CUDA', 'hybrid_cuda')
            ]:
                # Try to find the solver time for this method
                solver_time_match = re.search(
                    rf"{method} solver time: ([\d.]+) ms", content)
                
                if solver_time_match:
                    solver_time = float(solver_time_match.group(1))
                    
                    # Find iteration count if available
                    iterations_match = re.search(
                        rf"===== Running {method} Implementation =====[\s\S]*?Converged after (\d+) iterations", 
                        content)
                    iterations = int(iterations_match.group(1)) if iterations_match else None
                    
                    # Create a record for this implementation
                    data_record = {
                        'method': method_name,
                        'dataset': dataset_name,
                        'matrix_size': matrix_size,
                        'solver_time_ms': solver_time,
                        'iterations': iterations
                    }
                    all_data.append(data_record)
                    
                    # Save individual method performance CSV
                    method_csv = os.path.join(output_dir, f"{dataset_name}_{method_name}_performance.csv")
                    method_df = pd.DataFrame([data_record])
                    method_df.to_csv(method_csv, index=False)
    
    # Combine all data into a DataFrame
    if all_data:
        combined_df = pd.DataFrame(all_data)
        
        # Save combined dataset
        combined_csv = os.path.join(output_dir, "all_performance_data.csv")
        combined_df.to_csv(combined_csv, index=False)
        
        print(f"Combined performance data saved to {combined_csv}")
        return combined_df
    
    print("No performance data found in log files")
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
    
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Get unique datasets
    datasets = performance_data['dataset'].unique()
    
    # Extract sizes in MB from dataset names
    def extract_size(dataset_name):
        try:
            # Try to extract numeric portion from dataset name
            size_match = re.search(r'(\d+)mb', dataset_name.lower())
            if size_match:
                return int(size_match.group(1))
            else:
                # If not found, use the matrix_size as a fallback
                return performance_data[performance_data['dataset'] == dataset_name]['matrix_size'].iloc[0] / 10000  # Scale for display
        except:
            return 0
    
    # Map datasets to their sizes
    matrix_sizes = {dataset: extract_size(dataset) for dataset in datasets}
    
    # 1. Serial vs OpenMP
    if 'serial' in performance_data['method'].unique() and 'openmp' in performance_data['method'].unique():
        plt.figure(figsize=(12, 8))
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Sort data points by matrix size
        sorted_datasets = sorted(datasets, key=lambda x: matrix_sizes[x])
        
        # Get data for plotting
        serial_times = []
        openmp_times = []
        sizes = []
        
        for dataset in sorted_datasets:
            serial_data = performance_data[(performance_data['dataset'] == dataset) & 
                                          (performance_data['method'] == 'serial')]
            openmp_data = performance_data[(performance_data['dataset'] == dataset) & 
                                          (performance_data['method'] == 'openmp')]
            
            if not serial_data.empty and not openmp_data.empty:
                serial_times.append(serial_data['solver_time_ms'].iloc[0])
                openmp_times.append(openmp_data['solver_time_ms'].iloc[0])
                sizes.append(matrix_sizes[dataset])
        
        # Calculate average speedup
        if serial_times and openmp_times:
            speedups = [s/o for s, o in zip(serial_times, openmp_times)]
            avg_speedup = sum(speedups) / len(speedups)
            
            # Plot data
            plt.plot(sizes, serial_times, 'o-', color='blue', linewidth=2, label='Serial Time')
            plt.plot(sizes, openmp_times, 'o-', color='red', linewidth=2, label='OpenMP Time')
            
            # Add annotation for average speedup
            plt.annotate(f'Average Speedup: {avg_speedup:.2f}x', 
                        xy=(0.7, 0.9), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
            
            plt.title('Serial vs OpenMP Performance Comparison')
            plt.xlabel('Matrix Size (MB)')
            plt.ylabel('Time (ms)')
            plt.legend()
            plt.savefig(os.path.join(plots_dir, "serial_vs_openmp.png"), dpi=300)
            plt.close()
            
            print(f"Created Serial vs OpenMP plot")
    
    # 2. Serial vs Standard CUDA
    if 'serial' in performance_data['method'].unique() and 'standard_cuda' in performance_data['method'].unique():
        plt.figure(figsize=(12, 8))
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Sort data points by matrix size
        sorted_datasets = sorted(datasets, key=lambda x: matrix_sizes[x])
        
        # Get data for plotting
        serial_times = []
        cuda_times = []
        sizes = []
        
        for dataset in sorted_datasets:
            serial_data = performance_data[(performance_data['dataset'] == dataset) & 
                                          (performance_data['method'] == 'serial')]
            cuda_data = performance_data[(performance_data['dataset'] == dataset) & 
                                         (performance_data['method'] == 'standard_cuda')]
            
            if not serial_data.empty and not cuda_data.empty:
                serial_times.append(serial_data['solver_time_ms'].iloc[0])
                cuda_times.append(cuda_data['solver_time_ms'].iloc[0])
                sizes.append(matrix_sizes[dataset])
        
        # Calculate average speedup
        if serial_times and cuda_times:
            speedups = [s/c for s, c in zip(serial_times, cuda_times)]
            avg_speedup = sum(speedups) / len(speedups)
            
            # Plot data
            plt.plot(sizes, serial_times, 'o-', color='blue', linewidth=2, label='Serial Time')
            plt.plot(sizes, cuda_times, 'o-', color='red', linewidth=2, label='Standard CUDA Time')
            
            # Add annotation for average speedup
            plt.annotate(f'Average Speedup: {avg_speedup:.2f}x', 
                        xy=(0.7, 0.9), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
            
            plt.title('Serial vs Standard CUDA Performance Comparison')
            plt.xlabel('Matrix Size (MB)')
            plt.ylabel('Time (ms)')
            plt.legend()
            plt.savefig(os.path.join(plots_dir, "serial_vs_standard_cuda.png"), dpi=300)
            plt.close()
            
            print(f"Created Serial vs Standard CUDA plot")
    
    # 3. Serial vs Hybrid CUDA (exactly like your example)
    if 'serial' in performance_data['method'].unique() and 'hybrid_cuda' in performance_data['method'].unique():
        plt.figure(figsize=(12, 8))
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Sort data points by matrix size
        sorted_datasets = sorted(datasets, key=lambda x: matrix_sizes[x])
        
        # Get data for plotting
        serial_times = []
        hybrid_times = []
        sizes = []
        
        for dataset in sorted_datasets:
            serial_data = performance_data[(performance_data['dataset'] == dataset) & 
                                          (performance_data['method'] == 'serial')]
            hybrid_data = performance_data[(performance_data['dataset'] == dataset) & 
                                           (performance_data['method'] == 'hybrid_cuda')]
            
            if not serial_data.empty and not hybrid_data.empty:
                serial_times.append(serial_data['solver_time_ms'].iloc[0])
                hybrid_times.append(hybrid_data['solver_time_ms'].iloc[0])
                sizes.append(matrix_sizes[dataset])
        
        # Calculate average speedup
        if serial_times and hybrid_times:
            speedups = [s/h for s, h in zip(serial_times, hybrid_times)]
            avg_speedup = sum(speedups) / len(speedups)
            
            # Plot data
            plt.plot(sizes, serial_times, 'o-', color='blue', linewidth=2, label='Serial Time')
            plt.plot(sizes, hybrid_times, 'o-', color='red', linewidth=2, label='Hybrid CUDA Time')
            
            # Add annotation for average speedup
            plt.annotate(f'Average Speedup: {avg_speedup:.2f}x', 
                        xy=(0.7, 0.9), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
            
            plt.title('Serial vs Hybrid CUDA Performance Comparison')
            plt.xlabel('Matrix Size (MB)')
            plt.ylabel('Time (ms)')
            plt.legend()
            plt.savefig(os.path.join(plots_dir, "serial_vs_hybrid_cuda.png"), dpi=300)
            plt.close()
            
            print(f"Created Serial vs Hybrid CUDA plot")
    
    # 4. All implementations in one graph
    plt.figure(figsize=(12, 8))
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Sort data points by matrix size
    sorted_datasets = sorted(datasets, key=lambda x: matrix_sizes[x])
    
    # Plot data for each implementation
    methods = performance_data['method'].unique()
    colors = {'serial': 'blue', 'openmp': 'green', 'standard_cuda': 'red', 'hybrid_cuda': 'purple'}
    
    x_sizes = []
    data_by_method = {method: [] for method in methods}
    
    for dataset in sorted_datasets:
        size = matrix_sizes[dataset]
        x_sizes.append(size)
        
        for method in methods:
            method_data = performance_data[(performance_data['dataset'] == dataset) & 
                                           (performance_data['method'] == method)]
            
            if not method_data.empty:
                data_by_method[method].append((size, method_data['solver_time_ms'].iloc[0]))
    
    # Plot each method
    for method in methods:
        if data_by_method[method]:
            x_vals = [x for x, y in data_by_method[method]]
            y_vals = [y for x, y in data_by_method[method]]
            
            plt.plot(x_vals, y_vals, 'o-', color=colors.get(method, 'gray'), 
                     linewidth=2, label=f'{method.capitalize()} Time')
    
    plt.title('All Implementations Performance Comparison')
    plt.xlabel('Matrix Size (MB)')
    plt.ylabel('Time (ms)')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "all_implementations_comparison.png"), dpi=300)
    plt.close()
    
    print(f"Created All Implementations comparison plot")
    
    return plots_dir

def generate_summary_report(performance_data, convergence_data, output_dir):
    """Generate a summary report of all runs."""
    if performance_data is None or performance_data.empty:
        print("No performance data available for summary report")
        return
    
    summary_file = os.path.join(output_dir, "summary_report.md")
    
    try:
        from tabulate import tabulate
        has_tabulate = True
    except ImportError:
        print("Warning: tabulate package not found, using simple table formatting")
        has_tabulate = False
    
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
            matrix_size = dataset_data['matrix_size'].iloc[0] if 'matrix_size' in dataset_data.columns else 0
            
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
        if has_tabulate:
            f.write(tabulate(summary_rows, headers=headers, tablefmt="pipe"))
        else:
            # Simple table formatting
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
            for row in summary_rows:
                f.write("| " + " | ".join(str(cell) for cell in row) + " |\n")
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
        if has_tabulate:
            f.write(tabulate(speedup_rows, headers=headers, tablefmt="pipe"))
        else:
            # Simple table formatting
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
            for row in speedup_rows:
                f.write("| " + " | ".join(str(cell) for cell in row) + " |\n")
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
            if has_tabulate:
                f.write(tabulate(conv_rows, headers=headers, tablefmt="pipe"))
            else:
                # Simple table formatting
                f.write("| " + " | ".join(headers) + " |\n")
                f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
                for row in conv_rows:
                    f.write("| " + " | ".join(str(cell) for cell in row) + " |\n")
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

def interactive_dataset_selection(dataset_files):
    """Allow user to interactively select datasets to process."""
    print("\nAvailable datasets:")
    for i, dataset in enumerate(dataset_files):
        print(f"{i+1}. {os.path.basename(dataset)}")
    
    while True:
        selection = input("\nEnter dataset numbers to process (comma-separated, e.g., '1,3,5' or 'all' for all): ")
        if selection.lower() == 'all':
            return dataset_files
        
        try:
            indices = [int(x.strip()) - 1 for x in selection.split(',')]
            selected_datasets = [dataset_files[i] for i in indices if 0 <= i < len(dataset_files)]
            
            if not selected_datasets:
                print("No valid datasets selected. Please try again.")
                continue
            
            print("\nSelected datasets:")
            for dataset in selected_datasets:
                print(f"- {os.path.basename(dataset)}")
            
            confirm = input("\nConfirm selection? (y/n): ")
            if confirm.lower() in ('y', 'yes'):
                return selected_datasets
        except Exception as e:
            print(f"Invalid selection: {e}. Please try again.")

def interactive_thread_selection():
    """Allow user to interactively specify thread counts."""
    while True:
        selection = input("\nEnter thread counts to test (comma-separated, e.g., '1,2,4,8' or press Enter for default): ")
        
        if not selection.strip():
            print("Using default thread count from system.")
            return [None]
        
        try:
            thread_counts = [int(x.strip()) for x in selection.split(',')]
            
            if not all(t > 0 for t in thread_counts):
                print("Thread counts must be positive integers. Please try again.")
                continue
            
            print(f"\nSelected thread counts: {', '.join(str(t) for t in thread_counts)}")
            confirm = input("Confirm selection? (y/n): ")
            if confirm.lower() in ('y', 'yes'):
                return thread_counts
        except Exception as e:
            print(f"Invalid selection: {e}. Please try again.")

def interactive_implementation_selection():
    """Allow user to interactively select which implementations to run."""
    options = {
        '1': ('serial', 'Serial implementation'),
        '2': ('parallel', 'OpenMP parallel implementation'),
        '3': ('cuda', 'CUDA implementation'),
        '4': ('hybrid', 'Hybrid CUDA+OpenMP implementation'),
        '5': ('all', 'All implementations')
    }
    
    print("\nAvailable implementations:")
    for key, (_, desc) in options.items():
        print(f"{key}. {desc}")
    
    while True:
        selection = input("\nSelect implementation to run (1-5): ")
        
        if selection in options:
            choice = options[selection][0]
            print(f"Selected: {options[selection][1]}")
            
            # Return the appropriate flags
            if choice == 'serial':
                return True, False, False, False, True  # serial_only, parallel_only, cuda_only, hybrid_only, no_cuda
            elif choice == 'parallel':
                return False, True, False, False, True  # serial_only, parallel_only, cuda_only, hybrid_only, no_cuda
            elif choice == 'cuda':
                return False, False, True, False, False  # serial_only, parallel_only, cuda_only, hybrid_only, no_cuda
            elif choice == 'hybrid':
                return False, False, False, True, False  # serial_only, parallel_only, cuda_only, hybrid_only, no_cuda
            else:  # all
                return False, False, False, False, False  # serial_only, parallel_only, cuda_only, hybrid_only, no_cuda
        else:
            print("Invalid selection. Please choose a number between 1 and 5.")

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
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode to select datasets and thread counts')
    parser.add_argument('--skip-solver', action='store_true',
                        help='Skip running solver, just parse existing log files and generate plots')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all .mtx files in the datasets directory
    all_dataset_files = glob(os.path.join(args.datasets_dir, '*.mtx'))
    
    if not all_dataset_files:
        print(f"No .mtx files found in {args.datasets_dir}")
        return
    
    print(f"Found {len(all_dataset_files)} dataset files in the directory")
    
    # Skip solver execution if requested
    if args.skip_solver:
        print("\nSkipping solver execution as requested...")
        # Find existing log files
        log_files = glob(os.path.join(args.output_dir, "*_log.txt"))
        if not log_files:
            print("No log files found in the output directory. Cannot proceed.")
            return
            
        print(f"Found {len(log_files)} existing log files")
    else:
        # Interactive mode
        if args.interactive:
            print("\n===== Interactive Mode =====")
            
            # Let user select datasets
            dataset_files = interactive_dataset_selection(all_dataset_files)
            
            # Let user select thread counts
            thread_counts = interactive_thread_selection()
            
            # Let user select implementation
            serial_only, parallel_only, cuda_only, hybrid_only, no_cuda = interactive_implementation_selection()
            
            print("\nSummary of selections:")
            print(f"- Datasets: {', '.join(os.path.basename(f) for f in dataset_files)}")
            print(f"- Thread counts: {', '.join(str(t) if t else 'default' for t in thread_counts)}")
            implementation_str = "All implementations"
            if serial_only:
                implementation_str = "Serial only"
            elif parallel_only:
                implementation_str = "OpenMP parallel only"
            elif cuda_only:
                implementation_str = "CUDA only"
            elif hybrid_only:
                implementation_str = "Hybrid CUDA+OpenMP only"
            elif no_cuda:
                implementation_str = "CPU implementations only (no CUDA)"
            print(f"- Implementation: {implementation_str}")
            
            # Confirm before proceeding
            confirm = input("\nStart processing with these settings? (y/n): ")
            if confirm.lower() not in ('y', 'yes'):
                print("Aborted by user.")
                return
        else:
            # Non-interactive mode - use command line arguments
            dataset_files = all_dataset_files
            serial_only = args.serial_only
            parallel_only = args.parallel_only
            cuda_only = args.cuda_only
            hybrid_only = args.hybrid_only
            no_cuda = args.no_cuda
            
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
                    'serial_only': serial_only,
                    'parallel_only': parallel_only,
                    'cuda_only': cuda_only,
                    'hybrid_only': hybrid_only,
                    'no_cuda': no_cuda
                })
        
        print(f"\nPrepared {len(run_configs)} run configurations to process")
        
        # Execute runs sequentially
        results = []
        for config in run_configs:
            try:
                results.append(run_solver(**config))
            except KeyboardInterrupt:
                print("\nOperation interrupted by user. Processing available results...")
                break
        
        if not results:
            print("No results to process. Exiting.")
            return
        
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
        
        # Get log files from successful runs
        log_files = [r['log_file'] for r in successful_runs]
    
    # Extract performance data from log files
    print("\nExtracting performance data from log files...")
    performance_data = extract_performance_data_from_logs(log_files, args.output_dir)
    
    # Extract convergence information from log files
    print("Extracting convergence information from log files...")
    convergence_data = extract_convergence_info(log_files)
    
    # Generate plots
    print("Generating performance plots...")
    plots_dir = generate_plots(performance_data, args.output_dir)
    
    # Generate summary report
    print("Generating summary report...")
    summary_file = generate_summary_report(performance_data, convergence_data, args.output_dir)
    
    print(f"\nAnalysis complete! Check {args.output_dir} for results.")
    print(f"Summary report: {summary_file}")
    print(f"Performance plots: {os.path.join(args.output_dir, 'plots')}")

if __name__ == "__main__":
    main()
