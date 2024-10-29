#!/bin/bash

# first argument: number of qubits
# second argument: seed

# run experiments with all different qubits
for n_qubits in 3 4 5 6 7 8 9
do
    echo "Running with $n_qubits qubits"
    if [ -z "$1" ]
    then
        kedro run --params=n_qubits=$n_qubits &
    else
        if [ -z "$2" ]
        then
            kedro run --pipeline training --params=omegas=$n_qubits,n_qubits=$n_qubits,circuit_type=$1 &
        else
            kedro run --pipeline training --params=omegas=$n_qubits,n_qubits=$n_qubits,circuit_type=$1,seed=$2 &
        fi
    fi
    sleep 10
done