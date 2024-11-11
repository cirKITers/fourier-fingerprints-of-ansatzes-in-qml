#!/bin/bash

# first argument: number of qubits
# second argument: seed

# run experiments with all different qubits
for n_qubits in 2
do
    echo "Running with $n_qubits qubits"
    if [ -z "$1" ]
    then
        kedro run --params=n_qubits=$n_qubits &
    else
        # kedro run --pipeline training --params=omegas=$n_qubits,n_qubits=$n_qubits,circuit_type=$1,seed=$2 &
        sbatch --job-name "q$n_qubits-c$1-s$2" ./slurm_job.sh "omegas=$n_qubits,n_qubits=$n_qubits,circuit_type=$1,seed=$2" &
    fi
done