#!/bin/bash

# first argument: number of qubits
# second argument: seed

# run experiments with all different qubits
for n_qubits in 3 4 5 6
do
    echo "Running with $n_qubits qubits"
    if [ -z "$1" ]
    then
        kedro run --params=n_qubits=$n_qubits &
    else
        # kedro run --pipeline training --params=omegas=$n_qubits,n_qubits=$n_qubits,circuit_type=$1,seed=$2 &
        sbatch --job-name "q$n_qubits-c$1-s$2" ./slurm_job.sh "data.omegas=$n_qubits,model.n_qubits=$n_qubits,model.circuit_type=$1,seed=$2" &
        # sbatch --job-name "q$n_qubits-c$1" ./slurm_job.sh "data.omegas=$n_qubits,model.n_qubits=$n_qubits,model.circuit_type=$1" &
    fi
done