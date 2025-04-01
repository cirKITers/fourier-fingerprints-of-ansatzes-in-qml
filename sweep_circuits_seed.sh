#!/bin/bash

# first argument: seed

# run experiments with all different circuits
# for circuit in Hardware_Efficient Circuit_YZY_Entangling Circuit_YZY Circuit_19 Circuit_15 Circuit_17
for circuit in Hardware_Efficient Circuit_YZY_Entangling Circuit_YZY Circuit_19 Circuit_15 Circuit_17
do
    echo "Running with Ansatz $circuit"

    # use together with training and coefficients pipeline
    ./sweep_qubits_circuits_seed.sh $circuit $1
    # ./sweep_qubits_circuits_seed.sh $circuit

    # use together with coeffexpr pipeline
    # ./sweep_training_seed.sh $circuit $1
done