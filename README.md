# PLATO-MO-ELF
A python code for PLATO molecular orbital (MO) and electron localization function (ELF) calculation
# USAGE
Before using this code, ensure that you have 'numpy' and PLATO (A package of programs for building tight binding models) installed. 
 1. Run PLATO 'tb1' calculation to generate '*.wf' and '*.xyz' file.
 2. Copy '*.wf' and '*.xyz' to the '00_inputdata/' folder.
 3. Modify the parameters in 'input.py' as needed, and run this code.
 4. The output will be saved in the '01_results/' folder, visualise the '.cube' file by VMD or other visualisation software.
# EXAMPLE
Some *.wf and *.xyz files has been pre-placed in the '00_inputdata/' folder. The default input file is configured for ELF calculation of a water molecule.
