#!/bin/bash

# run experiments with all different circuits
for circuit in Circuit_1 Circuit_6 Circuit_19 Bansatz Strongly_Entangling Hardware_Efficient 
do
    echo "Running with Ansatz $circuit"
    kedro run --params="circuit_type=$circuit"
done
