for circuit in Hardware_Efficient Circuit_YZY Circuit_YZY_Entangling Circuit_19 Circuit_15 Circuit_17 Circuit_16 Circuit_18
do
    echo "Running with Ansatz $circuit"
    poetry run kedro run --pipeline "visualize" --params=model.circuit_type=$circuit,model.n_qubits=4,model.draw=True
done