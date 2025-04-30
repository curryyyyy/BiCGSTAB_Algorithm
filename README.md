# Download datasets
pip install gdown

gdown --folder https://drive.google.com/drive/folders/1vwDd_2Uz6I7eIDbZ-SVv46BAyHgwdxbc


# Before run the code:
1. Set thread size
Project properties > Configuration properties > Debugging > Command Arguments 
Enter : --threads 16
Click Apply.

2. Enable CUDA
Download and install the NVIDIA CUDA Toolkit if it is not already installed.
Right click project > Build dependencies > Build customizations > tick the CUDA > click OK.
Right click on project > Properties > Navigate to "CUDA C/C++" > Command Line > add this to Additional Options: "-allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"
Click Apply and Ok

3. Enable OpenMP
Project properties > C/C++ > Language > Enable Open MP Support

# Command to run python script:
pip install pandas

pip install matplotlib 

pip install tabulate


# Run tests with interactive mode
python updated-script-with-output.py --exe [exe path] --datasets-dir [dataset folder] --threads 16 --output-dir bicgstab_results --interactive 

# Run all datasets together to see all the performance visualizations
python updated-script-with-output.py --exe [exe path] --datasets-dir [dataset folder] --threads 16 --output-dir bicgstab_results
