#!/bin/bash

# run experiments with all different circuits
for seed in 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009
do
    for n_qubits in 4 5 6 7
    do
        for circuit in Circuit_1 Circuit_6 Circuit_19 Bansatz Strongly_Entangling Hardware_Efficient 
        do
            echo "Running with Ansatz $circuit"
            kedro run --params=circuit_type=$circuit,omegas=$n_qubits,n_qubits=$n_qubits,seed=$seed
        done
    done
done