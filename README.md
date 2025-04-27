Before run the code:
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
