## Temperature-dependent atomic cluster expansion interatomic potentials in LAMMPS
Modification to the performant atomic cluster expansion implementation in LAMMPS (https://github.com/ICAMS/lammps-user-pace) to allow for linear interpolation between ACE potentials fit at different temperatures. The code is messy as it was never fully tested after finishing my PhD; however, I hope that it may be of use to someone.

The below is the original README for the ML-PACE package:

The ML-PACE package provides the pace pair style,
an efficient implementation of the Atomic Cluster Expansion
potential (ACE).

ACE is a methodology for deriving a highly accurate classical
potential fit to a large archive of quantum mechanical (DFT) data.
This package was written by Yury Lysogorskiy and others
at ICAMS, the Interdisciplinary Centre for Advanced Materials Simulation,
Ruhr University Bochum, Germany (http://www.icams.de).

This package requires a library that can be downloaded and built
in lib/pace or somewhere else, which must be done before building
LAMMPS with this package. Details of the download, build, and
install process for this package using traditional make (not CMake)
are given in the lib/pace/README file, and scripts are
provided to help automate the process.  Also see the LAMMPS manual for
general information on building LAMMPS with external libraries
using either traditional make or CMake.

More information about the ML-PACE implementation of ACE
is available here:

https://github.com/ICAMS/lammps-user-pace
