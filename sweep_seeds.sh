#!/bin/bash

# run experiments with all different circuits
for seed in 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009
do
    echo "--- Seed $seed ---"
    # Quantum Model
    ./sweep_circuits_seed.sh $seed

    # Classical Model
    # sbatch --job-name "s$seed" ./slurm_job.sh "seed=$seed"
    # sleep 1.0
done