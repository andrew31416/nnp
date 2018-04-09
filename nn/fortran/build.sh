#!/bin/bash

FC="gfortran"
suffix="f95"
f2py="f2py"
f90wrap="f90wrap"

modname="nn_f95"

lapack_dir="/usr/lib/atlas-base/atlas/"

FFLAGS="-O2 -fPIC -llapack -lblas -lgomp -fopenmp"
DEBUG="-W -Wall -pedantic"

#--------------------------------------------#
# all files to compile .o and .mod files for #
#--------------------------------------------#

f1="config."
f2="feature_config."
f3="feature_util."
f4="io."
f5="util."
f6="init."
f7="propagate."
f8="measures."
f9="tapering."
f10="features."
f11="feature_selection."

#----------------------------------#
# files to create f2py pragma from #
#----------------------------------#

fwrap_files="features.f95 init.f95 util.f95 measures.f95 propagate.f95 feature_selection.f95"

#--------------------------------------#
# functions to take from wrapped files #
#--------------------------------------#

fwrap_functions="calculate_distance_distributions"
fwrap_functions+=" calculate_features_singleset"
fwrap_functions+=" initialise_net"
fwrap_functions+=" init_configs_from_disk"
fwrap_functions+=" init_features_from_disk"
fwrap_functions+=" init_feature_vectors"
fwrap_functions+=" init_loss"
fwrap_functions+=" num_threads"
fwrap_functions+=" loss"
fwrap_functions+=" loss_jacobian"
fwrap_functions+=" backprop_all_forces"
fwrap_functions+=" check_features"
fwrap_functions+=" check_feature_derivatives"
fwrap_functions+=" get_config"
fwrap_functions+=" get_nconf"
fwrap_functions+=" get_natm"
fwrap_functions+=" get_total_natm"
fwrap_functions+=" get_features"
fwrap_functions+=" get_num_nodes"
fwrap_functions+=" get_node_distribution"
fwrap_functions+=" get_ref_energies"
fwrap_functions+=" loss_feature_jacobian"

# clear previous build
rm $f1"o" $f1"mod"
rm $f2"o" $f2"mod"
rm $f3"o" $f3"mod"
rm $f4"o" $f4"mod"
rm $f5"o" $f5"mod"
rm $f6"o" $f6"mod"
rm $f7"o" $f7"mod"
rm $f8"o" $f8"mod"
rm $f9"o" $f9"mod"
rm $f10"o" $f10"mod"
rm $f11"o" $f11"mod"
rm f90wrap_*.f90
rm unittest.o unittest.mod

# initial build
$FC -c  $f1$suffix $FFLAGS $DEBUG 
$FC -c  $f2$suffix $FFLAGS $DEBUG
$FC -c  $f3$suffix $FFLAGS $DEBUG
$FC -c  $f4$suffix $FFLAGS $DEBUG
$FC -c  $f5$suffix $FFLAGS $DEBUG
$FC -c  $f6$suffix $FFLAGS $DEBUG
$FC -c  $f7$suffix $FFLAGS $DEBUG
$FC -c  $f8$suffix $FFLAGS $DEBUG
$FC -c  $f9$suffix $FFLAGS $DEBUG
$FC -c $f10$suffix $FFLAGS $DEBUG
$FC -c $f11$suffix $FFLAGS $DEBUG

$f90wrap -m $modname $fwrap_files -k kind_map -S 12 --only $fwrap_functions


$f2py -c -m $modname -L$lapack_dir -llapack -lblas f90wrap_*.f90 *.o --f90flags="-fPIC -fopenmp -llapack -lblas" -lgomp --fcompiler=$FC 
