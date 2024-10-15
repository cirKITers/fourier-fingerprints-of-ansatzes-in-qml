#!/bin/bash

# first argument: number of qubits
# second argument: seed

# run experiments with all different circuits
for circuit in Circuit_1 Circuit_6 Circuit_19 Bansatz Strongly_Entangling Hardware_Efficient 
do
    for n_qubits in 2 3 4 5
    do
        kedro run --pipeline coefficients --params=n_qubits=$n_qubits,circuit_type=$circuit &
        sleep 10
    done
done