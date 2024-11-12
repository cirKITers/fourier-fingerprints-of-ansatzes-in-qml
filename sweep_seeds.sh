#!/bin/bash

# run experiments with all different circuits
for seed in 1006 1007 1008
do
    echo "Running with seed $seed"
    ./sweep_circuits_seed.sh $seed
done