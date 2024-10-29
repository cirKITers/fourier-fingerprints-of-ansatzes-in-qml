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
        # kedro run --pipeline training --params=omegas=$n_qubits,n_qubits=$n_qubits,circuit_type=$1,seed=$2 &
        ./slurm_submit $n_qubits $1 $2 &
    fi
    sleep 10
done