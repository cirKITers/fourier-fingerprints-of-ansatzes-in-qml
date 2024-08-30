#!/bin/bash

# run experiments with all different circuits
for n_qubits in 3 4 5 6 7 8 9 10
do
    for circuit in Circuit_1 Circuit_6 Circuit_19 Bansatz Strongly_Entangling Hardware_Efficient 
    do
        echo "Running with Ansatz $circuit"
        kedro run --params=circuit_type=$circuit,omegas="$n_qubits",n_qubits=$n_qubits
    done
done