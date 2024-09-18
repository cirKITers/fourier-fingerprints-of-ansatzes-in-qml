#!/bin/bash

# run experiments with all different circuits
for n_qubits in 3 4 5 6 7 8 9 10
do
    sweep_circuits.sh $n_qubits
done