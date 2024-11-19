#!/bin/bash

# first argument: seed

# run experiments with all different circuits
for circuit in Circuit_YZY_N Circuit_19_N Bansatz_N Strongly_Entangling Hardware_Efficient_N
do
    echo "Running with Ansatz $circuit"

    ./sweep_qubits_circuits_seed.sh $circuit $1
done