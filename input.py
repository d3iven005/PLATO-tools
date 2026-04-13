job = 1   ### 1 for molecular orbital calculation, 2 for electron localisation function
energylevel = '37,38'   ## If job = 1, choose the orbital you want to calculate
jobname = 'Caffeine'
ncpu = 12
ifcrystal = 0  ## 0 for non-periodic, 1 for crystal
padding_distance = 3 ## The box boundary
vectorflag = 1 ## 0 for reading from xyz file, 1 for setting up by your self
vectorA = [20, 0, 0]
vectorB = [0, 20, 0]
vectorC = [0, 0, 20]  ##unit: Bohr = 0.529177 Angstrom
N1 = 150
N2 = 150
N3 = 150  ### number of grids for three vectors
