#!/bin/bash

# run experiments with all different circuits
for seed in 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009
do
    echo "Running with seed $seed"
    ./sweep_circuits_seed.sh $seed
done