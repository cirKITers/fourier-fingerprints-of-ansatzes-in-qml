#!/bin/bash

# first argument: seed

# run experiments with all different circuits
# for circuit in Hardware_Efficient Circuit_YZY_Entangling Circuit_YZY Circuit_19 Circuit_15 Circuit_17
for circuit in Hardware_Efficient Circuit_YZY_Entangling Circuit_YZY Circuit_19 Circuit_18 Circuit_17 Circuit_16 Circuit_15 
do
    echo "--- Ansatz $circuit ---"

    # with seed for training
    # ./sweep_qubits_circuits_seed.sh $circuit $1
    # without seed for coefficient and expressibility
    ./sweep_qubits_circuits_seed.sh $circuit

done