from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    calculate_coefficients,
    sample_coefficients,
    correlate,
    normalize,
    sweep_control_values,
    calculate_decay,
    weight_coefficients,
)


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                func=calculate_coefficients,
                inputs={
                    "model": "model",
                    "samples": "params:coefficients.samples",
                    "noise_params": "params:model.noise_params",
                    "seed": "params:seed",
                },
                outputs="coefficients",
                name="calculate_coefficients",
            ),
            node(
                func=sweep_control_values,
                inputs={
                    "model": "model",
                    "samples": "params:coefficients.samples",
                    "noise_params": "params:model.noise_params",
                    "seed": "params:seed",
                    "n_control_values": "params:n_control_values",
                },
                outputs="coefficients_correlated_control",
                name="sweep_control_values",
            ),
            node(
                func=calculate_decay,
                inputs={
                    "df": "coefficients",
                },
                outputs="coefficients_decay",
                name="calculate_decay",
            ),
            node(
                func=correlate,
                inputs={
                    "df": "coefficients",
                    "method": "params:coefficients.correlation_method",
                },
                outputs="coefficients_correlated",
                name="correlate_coefficients",
            ),
            node(
                func=normalize,
                inputs={
                    "df": "coefficients_correlated",
                },
                outputs="coefficients_correlated_normalized",
                name="normalize_coefficients",
            ),
            node(
                func=weight_coefficients,
                inputs={
                    "coefficients_correlated": "coefficients_correlated_normalized",
                    "coefficients_decay": "coefficients_decay",
                    "linear": "params:coefficients.linear_decay",
                },
                outputs="coefficients_correlated_weighted_normalized",
                name="weight_coefficients",
            ),
        ]
    )


def create_randcoeffs_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                func=sample_coefficients,
                inputs={
                    "model": "model",
                    "samples": "params:coefficients.samples",
                    "seed": "params:seed",
                },
                outputs="random_coefficients",
                name="sample_coefficients",
            ),
            node(
                func=correlate,
                inputs={
                    "df": "random_coefficients",
                    "method": "params:coefficients.correlation_method",
                },
                outputs="random_coefficients_correlated",
                name="correlate_random_coefficients",
            ),
            node(
                func=normalize,
                inputs={
                    "df": "random_coefficients_correlated",
                },
                outputs="random_coefficients_correlated_normalized",
                name="normalize_random_coefficients",
            ),
        ]
    )
