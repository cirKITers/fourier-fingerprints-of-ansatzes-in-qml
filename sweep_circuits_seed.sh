#!/bin/bash

# first argument: seed

# run experiments with all different circuits
for circuit in Circuit_XZX_N Circuit_19_N Bansatz_N Strongly_Entangling Hardware_Efficient_N 
do
    ./sweep_qubits_circuits_seed.sh $circuit $1
done