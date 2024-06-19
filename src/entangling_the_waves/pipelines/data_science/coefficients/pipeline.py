from kedro.pipeline import Pipeline, node, pipeline

from .nodes import calculate_coefficients, correlate, normalize


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                func=calculate_coefficients,
                inputs={
                    "model": "model",
                    "samples": "params:samples",
                    "noise_params": "params:noise_params",
                    "seed": "params:seed",
                },
                outputs="coefficients",
                name="calculate_coefficients",
            ),
            node(
                func=correlate,
                inputs={
                    "df": "coefficients",
                },
                outputs="coefficients_correlated",
                name="correlate",
            ),
            node(
                func=normalize,
                inputs={
                    "df": "coefficients_correlated",
                },
                outputs="coefficients_correlated_normalized",
                name="normalize",
            ),
        ]
    )
