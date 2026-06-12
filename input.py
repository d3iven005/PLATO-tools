job = 2   ### 1 for molecular orbital calculation, 2 for electron localisation function
energylevel = '37,38'   ## If job = 1, choose the orbital you want to calculate
jobname = 'MGO'
ncpu = 12
level_chunk_size = 4  ## number of occupied energy levels calculated together in ELF; larger uses more memory
ifcrystal = 1  ## 0 for non-periodic, 1 for crystal
padding_distance = 3 ## The box boundary
use_local_box = 1  ## True for calculating only a local orthogonal output box
box_xmin = 1.0
box_xmax = 3.0
box_ymin = 1.0
box_ymax = 3.0
box_zmin = 1.0
box_zmax = 3.0  ## local box bounds, unit: Angstrom
local_box_normalize_orbitals = False  ## local boxes should usually use PLATO/AO normalization directly
vectorflag = 0 ## 0 for reading from xyz file, 1 for setting up by your self
vectorA = [20, 0, 0]
vectorB = [0, 20, 0]
vectorC = [0, 0, 20]  ##unit: Bohr = 0.529177 Angstrom
N1 = 50
N2 = 50
N3 = 50  ### number of grids for three vectors
