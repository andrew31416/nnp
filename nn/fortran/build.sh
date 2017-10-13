#!/bin/bash

FC="gfortran"
suffix="f95"
f2py="f2py"
f90wrap="f90wrap"

modname="nn_f95"

FFLAGS="-O2 -fPIC"
DEBUG="-fcheck=all -W -Wall -pedantic"

f1="config."$suffix
f2="io."$suffix
f3="init."$suffix

# initial build
$FC -c $f1 $FFLAGS $DEBUG 
$FC -c $f2 $FFLAGS $DEBUG
$FC -c $f3 $FFLAGS $DEBUG

#$f90wrap -m $modname $f1 $f2 $f3 -k kind_map -S 12 

$f2py -c -m $modname f90wrap_*.f90 *.o --f90flags="-fPIC -fopenmp" -lgomp --fcompiler=$FC
