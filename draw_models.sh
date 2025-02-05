for circuit in Circuit_YZY Circuit_YZY_Entangling Circuit_19 Hardware_Efficient Circuit_15 Circuit_1 Circuit_2 Circuit_16 Circuit_17
do
    echo "Running with Ansatz $circuit"
    poetry run kedro run --pipeline "visualize" --params=model.circuit_type=$circuit,model.n_qubits=4,model.draw=True
done