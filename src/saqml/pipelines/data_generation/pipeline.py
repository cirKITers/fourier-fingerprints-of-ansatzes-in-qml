from kedro.pipeline import Pipeline, node, pipeline

from .nodes import sample_domain, generate_fourier_series, create_model, print_model


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                func=create_model,
                inputs={
                    "n_qubits": "params:model.n_qubits",
                    "n_layers": "params:model.n_layers",
                    "circuit_type": "params:model.circuit_type",
                    "data_reupload": "params:model.data_reupload",
                    "initialization": "params:model.initialization",
                    "initialization_domain": "params:model.initialization_domain",
                    "output_qubit": "params:model.output_qubit",
                    "seed": "params:seed",
                },
                outputs="model",
                name="create_model",
            ),
            node(
                func=print_model,
                inputs={
                    "model": "model",
                },
                outputs="model_str",
                name="print_model",
            ),
            node(
                func=sample_domain,
                inputs={
                    "domain": "params:data.domain",
                    "omegas": "params:data.omegas",
                },
                outputs="domain_samples",
                name="sample_domain",
            ),
            node(
                func=generate_fourier_series,
                inputs={
                    "domain_samples": "domain_samples",
                    "omegas": "params:data.omegas",
                    "coefficients": "params:data.amplitude",
                },
                outputs="fourier_series",
                name="generate_fourier_series",
            ),
        ]
    )
