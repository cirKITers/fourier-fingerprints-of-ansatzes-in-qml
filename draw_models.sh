for circuit in Hardware_Efficient Circuit_15 Circuit_16 Circuit_17 Circuit_18 Circuit_19 Circuit_YZY Circuit_YZY_Entangling
do
    echo "Running with Ansatz $circuit"
    poetry run kedro run --pipeline "visualize" --params=model.circuit_type=$circuit,model.n_qubits=4,model.draw=True
done