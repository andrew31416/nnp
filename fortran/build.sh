#!/bin/bash

#--------------------------------------------------------------#
# Make script for fortran modules wrapped by Python interfaces #
#                                                              #
# See README for requirements and install guide                #
#--------------------------------------------------------------#

# debug options
debug="-fcheck=all"

# fortran compiler
FC="gfortran"

# f2py binary name
f2py="f2py"

# f90wrap binary name
f90wrap="f90wrap"

modname="assorted"

filename1="bond_generation.f90"
filename2="rvm_basis.f90"
filename3="util.f90"
filename4="sedc.f90"
filename5="io.f90"

$FC -c -W -Wall -pedantic $filename1 -fPIC -O2 $debug
$FC -c -W -Wall -pedantic $filename2 -fPIC -O2 -fopenmp -lgomp $debug
$FC -c -W -Wall -pedantic $filename3 -fPIC -O2 -fopenmp -lgomp $debug
$FC -c -W -Wall -pedantic $filename4 -fPIC -O2 -fopenmp -lgomp $debug
$FC -c -W -Wall -pedantic $filename5 -fPIC -O2 $debug

$f90wrap -m $modname $filename1 $filename2 $filename3 $filename4 $filename5 -k kind_map -S 12 

# include OpenMP routines
$f2py -c -m $modname f90wrap_*.f90 *.o --f90flags="-fPIC -fopenmp" -lgomp --fcompiler=$FC 
