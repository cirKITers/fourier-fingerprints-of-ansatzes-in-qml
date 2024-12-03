from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    visualize_coefficients_correlated,
    visualize_parameters_correlated,
    visualize_parameters_coefficients_correlated,
    visualize_coefficients_correlated_control,
    visualize_model,
)


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                func=visualize_coefficients_correlated,
                inputs={
                    "df": "coefficients_correlated_normalized",
                    "model": "model",
                    "positive_only": "params:training.positive_coeffs_only",
                },
                outputs="fig_coefficients_correlated",
                name="visualize_coefficients_correlated",
            ),
            node(
                func=visualize_coefficients_correlated,
                inputs={
                    "df": "random_coefficients_correlated_normalized",
                    "model": "model",
                    "positive_only": "params:training.positive_coeffs_only",
                },
                outputs="fig_random_coefficients_correlated",
                name="visualize_random_coefficients_correlated",
            ),
            node(
                func=visualize_parameters_correlated,
                inputs={
                    "df": "coefficients_correlated_normalized",
                    "model": "model",
                },
                outputs="fig_parameters_correlated",
                name="visualize_parameters_correlated",
            ),
            node(
                func=visualize_parameters_coefficients_correlated,
                inputs={
                    "df": "coefficients_correlated_normalized",
                    "model": "model",
                },
                outputs="fig_parameters_coefficients_correlated",
                name="visualize_parameters_coefficients_correlated",
            ),
            node(
                func=visualize_coefficients_correlated_control,
                inputs={
                    "df": "coefficients_correlated_control",
                    "model": "model",
                },
                outputs="fig_coefficients_correlated_control",
                name="visualize_coefficients_correlated_control",
            ),
        ]
    )


def create_model_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                func=visualize_model,
                inputs={
                    "model": "trained_model",
                    "domain_samples": "domain_samples",
                    "fourier_series": "fourier_series",
                    "noise_params": "params:model.noise_params",
                },
                outputs={"model": "fig_model_trained"},
                name="visualize_model_trained",
            ),
            node(
                func=visualize_model,
                inputs={
                    "model": "model",
                    "domain_samples": "domain_samples",
                    "fourier_series": "fourier_series",
                    "noise_params": "params:model.noise_params",
                },
                outputs={"model": "fig_model_initial"},
                name="visualize_model_initial",
            ),
        ]
    )
