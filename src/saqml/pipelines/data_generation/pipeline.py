from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    sample_domain,
    generate_fourier_series,
    create_model,
    print_model,
    get_fourier_dataset,
    get_hep_dataset,
    calculate_hep_spectrum,
)


def create_model_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                func=create_model,
                inputs={
                    "n_qubits": "params:model.n_qubits",
                    "n_layers": "params:model.n_layers",
                    "circuit_type": "params:model.circuit_type",
                    "data_reupload": "params:model.data_reupload",
                    "encoding": "params:model.encoding",
                    "initialization": "params:model.initialization",
                    "initialization_domain": "params:model.initialization_domain",
                    "mp_threshold": "params:model.mp_threshold",
                    "output_qubit": "params:model.output_qubit",
                    "seed": "params:seed",
                    "layer_multiplier": "params:model.layer_multiplier",
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
        ]
    )


def create_fourier_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                func=sample_domain,
                inputs={
                    "domain": "params:data.fourier.domain",
                    "omegas": "params:data.fourier.omegas",
                },
                outputs="domain_samples",
                name="sample_domain",
            ),
            node(
                func=generate_fourier_series,
                inputs={
                    "domain_samples": "domain_samples",
                    "omegas": "params:data.fourier.omegas",
                    "coefficients_mean": "params:data.fourier.coefficients.mean",
                    "coefficients_variance": "params:data.fourier.coefficients.variance",
                    "coefficients_distribution": "params:data.fourier.coefficients.distribution",
                    "offset": "params:data.fourier.offset",
                    "seed": "params:data.seed",
                },
                outputs={
                    "fourier_series": "fourier_series",
                    "target": "coeffs_target",
                },
                name="generate_fourier_series",
            ),
            node(
                func=get_fourier_dataset,
                inputs={
                    "batch_size": "params:training.batch_size",
                    "domain_samples": "domain_samples",
                    "fourier_series": "fourier_series",
                    # "coeffs_target": "coeffs_target",
                },
                outputs={
                    "train_loader": "train_loader",
                    "valid_loader": "valid_loader",
                },
                name="get_fourier_dataset",
            ),
        ]
    )


def create_hep_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
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
            node(
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
