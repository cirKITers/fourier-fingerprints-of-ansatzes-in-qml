#!/bin/bash

# first argument: number of qubits
# second argument: seed

# run experiments with all different qubits
for layer_multiplier in 1 2 3 4 5
do
    echo "Running with $layer_multiplier add. layers"
    sbatch --job-name "l$layer_multiplier-q4-cML_Bansatz-s$1" ./slurm_job.sh "model.layer_multiplier=$layer_multiplier,model.circuit_type=ML_Bansatz,seed=$1" &
done