from kedro.pipeline import Pipeline, node, pipeline

from .nodes import train_model


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                func=train_model,
                inputs={
                    "model": "model",
                    "domain_samples": "domain_samples",
                    "fourier_series": "fourier_series",
                    "noise_params": "params:noise_params",
                    "epochs": "params:epochs",
                    "learning_rate": "params:learning_rate",
                    "batch_size": "params:batch_size",
                },
                outputs="trained_model",
                name="train_model",
            ),
        ]
    )
