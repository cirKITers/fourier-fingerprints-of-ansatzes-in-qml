from kedro.pipeline import Pipeline, node, pipeline

from .nodes import sample_domain, generate_fourier_series, create_model


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                func=create_model,
                inputs={
                    "n_qubits": "params:n_qubits",
                    "n_layers": "params:n_layers",
                    "circuit_type": "params:circuit_type",
                    "data_reupload": "params:data_reupload",
                    "output_qubit": "params:output_qubit",
                },
                outputs="model",
                name="create_model",
            ),
            node(
                func=sample_domain,
                inputs={
                    "domain": "params:domain",
                    "omegas": "params:omegas",
                },
                outputs="domain_samples",
                name="sample_domain",
            ),
            node(
                func=generate_fourier_series,
                inputs={
                    "domain_samples": "domain_samples",
                    "omegas": "params:omegas",
                    "coefficients": "params:coefficients",
                },
                outputs="fourier_series",
                name="generate_fourier_series",
            ),
        ]
    )
