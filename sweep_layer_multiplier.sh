#!/bin/bash

# first argument: number of qubits
# second argument: seed

# run experiments with all different qubits
for layer_multiplier in 3 4 5 6 7 8
do
    echo "Running with $layer_multiplier add. layers"
    sbatch --job-name "l$layer_multiplier-q5-cML_Bansatz-s1000" ./slurm_job.sh "data.omegas=5,model.layer_multiplier=$layer_multiplier,model.n_qubits=5,model.circuit_type=ML_Bansatz,seed=1000" &
done