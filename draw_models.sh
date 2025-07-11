for circuit in Strongly_Entangling Hardware_Efficient Circuit_1 Circuit_3 Circuit_16 Circuit_18 Circuit_9 Circuit_2 Circuit_17 Circuit_19 Circuit_15
do
    echo "Running with Ansatz $circuit"
    poetry run kedro run --pipeline "visualize" --params=model.circuit_type=$circuit,model.n_qubits=4,model.draw=True
done