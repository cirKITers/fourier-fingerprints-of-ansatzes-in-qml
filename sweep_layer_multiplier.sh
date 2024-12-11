#!/bin/bash

# first argument: number of qubits
# second argument: seed

# run experiments with all different qubits
for layer_multiplier in 1 2 3 4 5
do
    echo "Running with $layer_multiplier add. layers"
    sbatch --job-name "l$layer_multiplier-c$1-s$2" ./slurm_job.sh "model.layer_multiplier=$layer_multiplier,model.circuit_type=$1,seed=$2" &
done