#!/bin/bash

# first argument: number of qubits
# second argument: seed

# run experiments with all different qubits
for training_seed in 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009
do
    echo "Running with $training_seed training seed"
    sbatch --job-name "ts$training_seed-c$1-s$2" ./slurm_job.sh "data.coefficients.seed=$training_seed,model.circuit_type=$1,seed=$2" &
done