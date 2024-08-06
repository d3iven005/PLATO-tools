job = 2   ### 1 for molecular orbital calculation, 2 for electron localisation function
energylevel = '1,2,4-6'   ## If job = 1, choose the orbital you want to calculate
jobname = 'h2oy'
ncpu = 8
ifcrystal = 0  ## 0 for non-periodic, 1 for crystal
vectorflag = 0 ## 0 for reading from xyz file, 1 for setting up by your self
vectorA = [20, 0, 0]
vectorB = [0, 20, 0]
vectorC = [0, 0, 20]  ##unit: Bohr = 0.529177 Angstrom
N1 = 100
N2 = 100
N3 = 100  ### number of grids for three vectors