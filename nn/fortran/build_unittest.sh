#!/bin/bash

testname="unittest_bin"

FC="gfortran"

f_suffix="f95"
o_suffix="o"

FFLAGS="-O2 -llapack -lblas"
DEBUG="-fcheck=all -W -Wall -pedantic"

f1="config."
f2="io."
f3="init."
f4="propagate."
f5="util."
f6="measures."
f7="feature_config."
f8="feature_util."
f9="features."

unittest="unittest."

# clear previous build #
rm $f1"o"  $f2"o"


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

$FC -o $testname $unittest"f95" config.o init.o propagate.o util.o io.o measures.o feature_config.o feature_util.o features.o $FFLAGS $DEBUG


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
