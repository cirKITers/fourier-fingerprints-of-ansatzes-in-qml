from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    visualize_coefficients_correlated,
    visualize_parameters_correlated,
    visualize_parameters_coefficients_correlated,
)


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                func=visualize_coefficients_correlated,
                inputs={
                    "df": "coefficients_correlated_normalized",
                    "model": "model",
                },
                outputs="fig_coefficients_correlated",
                name="visualize_coefficients_correlated",
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
        ]
    )
