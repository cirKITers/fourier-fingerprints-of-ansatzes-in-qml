#!/bin/bash

# run experiments with all different circuits
for circuit in Circuit_1 Circuit_5 Circuit_9 Circuit_15 Circuit_18 Circuit_19 Hardware_Efficient Strongly_Entangling No_Entangling
do
    echo "Running $circuit"
    kedro run --params="circuit_type=$circuit"
done