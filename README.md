# PLATO-tools
A Python code for PLATO molecular orbital (MO), charge density, and electron localization function (ELF) calculation. If you have any questions about this tools, please feel free to contact me via jiwen.yu18@imperial.ac.uk.
# USAGE
Before using this code, ensure that you have Python 'numpy' package and PLATO (A package of programs for building tight binding models) installed. PLATO is a free program and for PLATO installation requests, please contact Prof. Andrew Horsfield via a.horsfield@imperial.ac.uk. 
 1. Run PLATO "tb1" calculation to generate "*.wf" and "*.xyz" file.
 2. Copy "*.wf" and "*.xyz" to the "00_inputdata/" folder. Some versions of PLATO for non-periodic calculation will generate ".wf" file without K-point information. Please copy "K-point 1   0.00000   0.00000   0.00000 1.0000000000" to the first line of ".wf" file.
 3. Modify the parameters in "input.py" as needed, and run this code.
 4. The output will be saved in the "01_results/" folder, visualise the "*.cube" file by VMD or other visualisation software.
# EXAMPLE
Some "*.wf" and "*.xyz" files have been pre-placed in the "00_inputdata/" folder. The default input file is configured for ELF calculation of a water molecule.
# PLATO DATASET USAGE
 1. Copy the directory "PLATO-tools/PLATOdataset/HCOMg_new" to the directory "Your_PLATO_route/Data/TightBinding".
 2. Change the 'Dataset' Flag in the input file to "HCOMg_new".
 3. Copy all the '.mdt' files in the directory "PLATO-tools/PLATOdataset/Multipole/SCF2(dipole)" or "PLATO-tools/PLATOdataset/Multipole/SCF3(quadrupole)" (not recommended)) to "Your_PLATO_route/Data/TightBinding/HCOMg_new".
 4. Change the 'SCFFlag' Flag in the input file to 2 for dipole, or 3 for quardrupole. Note that the '.mdt' file in 'SCF2(dipole)' only support monopole and dipole calculation, and 'SCF3(quadrupole)' only supports monopole and quadrupole calculation.
 5. Run the "tb1" program.
# NOTICE
The ELF code right now is calculated under its original definition in real space, which results in very low calculation efficiency. This calculation can be held in reciprocal space after Fourier transformation. The gradient of wavefunction phi and charge density will be equal to ik(Phi(k)) and ik(rho(k)), which will increase the calculation efficiency significantly. This will be done in the future.
Support for Gaussian and Slater-type orbitals will also be added in the future.
# NOTICE*
Note: Na, K, Ca, and Fe are still problematic in the "NewDataset/". Any hopping or overlap integrals involving these elements must be excluded.
