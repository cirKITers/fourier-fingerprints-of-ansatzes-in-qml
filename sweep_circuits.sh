#!/bin/bash

# first argument: number of qubits
# second argument: seed

# run experiments with all different circuits
for circuit in Circuit_1 Circuit_6 Circuit_19 Bansatz Strongly_Entangling Hardware_Efficient 
do
    echo "Running with Ansatz $circuit"
    if [ -z "$1" ]
    then
        kedro run --params=circuit_type=$circuit
    else
        if [ -z "$2" ]
        then
            kedro run --params=circuit_type=$circuit,omegas=$1,n_qubits=$1
        else
            kedro run --params=circuit_type=$circuit,omegas=$1,n_qubits=$1,seed=$2
        fi
    fi
done