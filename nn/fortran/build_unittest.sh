#!/bin/bash

testname="unittest_bin"

FC="gfortran"

f_suffix="f95"
o_suffix="o"

FFLAGS="-O2 -llapack -lblas -lgomp"
DEBUG="-fcheck=all -W -Wall -pedantic -fcheck=all"

f1="tapering."
f2="config."
f3="feature_config."
f4="util."
f5="feature_util."
f6="io."
f7="lookup."
f8="init."
f9="propagate."
f10="measures."
f11="features."
f12="feature_selection."

unittest="unittest."

#----------------------#
# clear previous build #
#----------------------#

rm $f1"o"  $f1"mod"
rm $f2"o"  $f2"mod"
rm $f3"o"  $f3"mod"
rm $f4"o"  $f4"mod"
rm $f5"o"  $f5"mod"
rm $f6"o"  $f6"mod"
rm $f7"o"  $f7"mod"
rm $f8"o"  $f8"mod"
rm $f9"o"  $f9"mod"
rm $f10"o" $f10"mod"
rm $f11"o" $f11"mod"
rm $f12"o" $f12"mod"

rm $testname

#------------------#
# build unit tests #
#------------------#

# initial build
$FC -c $f1$f_suffix $FFLAGS $DEBUG 
$FC -c $f2$f_suffix $FFLAGS $DEBUG
$FC -c $f3$f_suffix $FFLAGS $DEBUG
$FC -c $f4$f_suffix $FFLAGS $DEBUG
$FC -c $f5$f_suffix $FFLAGS $DEBUG
$FC -c $f6$f_suffix $FFLAGS $DEBUG
$FC -c $f7$f_suffix $FFLAGS $DEBUG
$FC -c $f8$f_suffix $FFLAGS $DEBUG
$FC -c $f9$f_suffix $FFLAGS $DEBUG
$FC -c $f10$f_suffix $FFLAGS $DEBUG
$FC -c $f11$f_suffix $FFLAGS $DEBUG
$FC -c $f12$f_suffix $FFLAGS $DEBUG

$FC -o $testname $unittest"f95" config.o init.o propagate.o util.o io.o measures.o feature_config.o feature_util.o tapering.o features.o feature_selection.o lookup.o $FFLAGS $DEBUG


#-----------#
# run tests #
#-----------#

"./"$testname

unittest_status=$?

if [ $unittest_status == 1 ]; then
    # SUCCESS
    exit 1
else
    # FAILURE
    exit 0
fi
