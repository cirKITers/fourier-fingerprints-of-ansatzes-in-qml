#!/bin/bash

# first argument: seed

# run experiments with all different circuits
# for circuit in Circuit_YZY_N Circuit_19_N Bansatz_N Strongly_Entangling Hardware_Efficient_N
for circuit in Hardware_Efficient Circuit_YZY_Entangling Circuit_YZY Circuit_19 Circuit_15 Circuit_17
do
    echo "Running with Ansatz $circuit"

    # use together with training and coefficients pipeline
    ./sweep_qubits_circuits_seed.sh $circuit $1

    # use together with coeffexpr pipeline
    # ./sweep_layer_multiplier.sh $circuit $1
done