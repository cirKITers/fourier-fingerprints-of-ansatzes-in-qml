#!/bin/bash

# first argument: number of qubits
# second argument: seed

# run experiments with all different qubits
for n_qubits in 6
do
    echo "--- $n_qubits qubits ---"
    if [ -z "$1" ]
    then
        kedro run --params=n_qubits=$n_qubits &
    else
        # with seed (training)
        # sbatch --job-name "q$n_qubits-c$1-s$2" ./slurm_job.sh "data.fourier.omegas=$n_qubits,model.n_qubits=$n_qubits,model.circuit_type=$1,seed=$2"
        # without seed (expressibility, coefficients)
        sbatch --job-name "q$n_qubits-c$1" ./slurm_job.sh "data.fourier.omegas=$n_qubits,model.n_qubits=$n_qubits,model.circuit_type=$1"
    fi
    sleep 1.0
done