from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    calculate_coefficients,
    filter_coefficients,
    sample_coefficients,
    correlate,
    correlate_complex,
    normalize,
    # sweep_control_values,
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
                    "n_samples": "params:coefficients.n_samples",
                    "noise_params": "params:model.noise_params",
                    "seed": "params:seed",
                },
                outputs="coefficients",
                name="calculate_coefficients",
            ),
            node(
                func=filter_coefficients,
                inputs={
                    "df": "coefficients",
                    "model": "model",
                },
                outputs="coefficients_filtered",
                name="filter_coefficients",
            ),
            # node(
            #     func=sweep_control_values,
            #     inputs={
            #         "model": "model",
            #         "n_samples": "params:coefficients.n_samples",
            #         "noise_params": "params:model.noise_params",
            #         "seed": "params:seed",
            #         "n_control_values": "params:n_control_values",
            #     },
            #     outputs="coefficients_correlated_control",
            #     name="sweep_control_values",
            # ),
            node(
                func=calculate_decay,
                inputs={
                    "df": "coefficients_filtered",
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
                func=correlate_complex,
                inputs={
                    "df": "coefficients",
                    "method": "params:coefficients.correlation_method",
                },
                outputs="coefficients_correlated_complex",
                name="correlate_complex_coefficients",
            ),
            node(
                func=filter_coefficients,
                inputs={
                    "df": "coefficients_correlated",
                    "model": "model",
                },
                outputs="coefficients_correlated_filtered",
                name="filter_coefficients_correlated",
            ),
            node(
                func=filter_coefficients,
                inputs={
                    "df": "coefficients_correlated_complex",
                    "model": "model",
                },
                outputs="coefficients_correlated_complex_filtered",
                name="filter_coefficients_correlated_complex",
            ),
            node(
                func=weight_coefficients,
                inputs={
                    "df": "coefficients_correlated_filtered",
                    "coefficients_decay": "coefficients_decay",
                    "weighting": "params:coefficients.weighting",
                },
                outputs="coefficients_correlated_filtered_weighted",
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
                    "n_samples": "params:coefficients.n_samples",
                    "seed": "params:data.seed",
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
                func=filter_coefficients,
                inputs={
                    "df": "random_coefficients_correlated",
                    "model": "model",
                },
                outputs="random_coefficients_correlated_filtered",
                name="filter_coefficients_correlated",
            ),
        ]
    )
