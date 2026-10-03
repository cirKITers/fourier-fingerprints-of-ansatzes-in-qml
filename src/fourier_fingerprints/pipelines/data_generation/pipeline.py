from kedro.pipeline import Pipeline, Node, pipeline

from .nodes import (
    generate_fourier_series,
    build_fourier_series_dataloader,
    generate_model,
    create_classical_model,
    tikz_model,
    print_model,
    get_hep_dataset,
    calculate_hep_spectrum,
)


def draw_model_pipeline() -> Pipeline:
    return pipeline(
        [
            Node(
                func=tikz_model,
                inputs={
                    "n_qubits": "params:model.n_qubits",
                    "n_layers": "params:model.n_layers",
                    "circuit_type": "params:model.circuit_type",
                    "output_qubit": "params:model.output_qubit",
                },
                outputs="model_str",
                name="tikz_model",
            ),
        ]
    )


def create_model_pipeline() -> Pipeline:
    return pipeline(
        [
            Node(
                generate_model,
                name="generate_model",
                tags=["generation"],
                inputs=[
                    "params:model.n_qubits",
                    "params:model.n_layers",
                    "params:model.circuit_type",
                    "params:model.data_reupload",
                    "params:model.encoding_gates",
                    "params:model.encoding_strategy",
                    "params:model.initialization",
                    "params:model.initialization_domain",
                    "params:model.output_qubit",
                    "params:model.seed",
                ],
                outputs={
                    "model": "model",
                },
            ),
            Node(
                func=print_model,
                inputs={
                    "model": "model",
                },
                outputs="model_str",
                name="print_model",
            ),
        ]
    )


def create_classical_model_pipeline() -> Pipeline:
    return pipeline(
        [
            Node(
                func=create_classical_model,
                inputs={
                    "width": "params:model.width",
                    "depth": "params:model.depth",
                    "seed": "params:seed",
                },
                outputs="model",
                name="create_model",
            ),
            Node(
                func=print_model,
                inputs={
                    "model": "model",
                },
                outputs="model_str",
                name="print_model",
            ),
        ]
    )


def create_fourier_pipeline() -> Pipeline:
    return pipeline(
        [
            Node(
                generate_fourier_series,
                name="generate_fourier_series",
                tags=["generation"],
                inputs=[
                    "model",
                    "params:data.coefficients_min",
                    "params:data.coefficients_max",
                    "params:data.zero_centered",
                    "params:data.seed",
                ],
                outputs={
                    "domain_samples": "domain_samples",
                    "fourier_samples": "fourier_samples",
                    "coefficients": "coefficients",
                },
            ),
            Node(
                build_fourier_series_dataloader,
                name="build_fourier_series_dataloader",
                tags=["generation"],
                inputs=[
                    "params:data.batch_size",
                    "domain_samples",
                    "fourier_samples",
                    "coefficients",
                ],
                outputs={
                    "train_loader": "train_loader",
                    "valid_loader": "valid_loader",
                },
            ),
        ]
    )


def create_hep_pipeline() -> Pipeline:
    return pipeline(
        [
            Node(
                func=get_hep_dataset,
                inputs={
                    "batch_size": "params:training.batch_size",
                    "n_events": "params:data.hep.n_events",
                    "features": "params:data.hep.features",
                    "scaling_methods": "params:data.hep.scaling_methods",
                    "labels": "params:data.hep.labels",
                    "seed": "params:data.seed",
                },
                outputs={
                    "train_loader": "train_loader",
                    "valid_loader": "valid_loader",
                    "scalers": "scalers",
                },
                name="get_hep_dataset",
            ),
            Node(
                func=calculate_hep_spectrum,
                inputs={
                    "data_loader": "train_loader",
                    "scalers": "scalers",
                    "model": "model",
                    "mts": "params:data.mts",
                    "mfs": "params:data.mfs",
                },
                outputs={
                    "target": "coeffs_target",
                },
                name="calculate_hep_spectrum",
            ),
        ]
    )
