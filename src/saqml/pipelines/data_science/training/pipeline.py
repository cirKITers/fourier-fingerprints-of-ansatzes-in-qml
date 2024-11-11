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
                    "noise_params": "params:model.noise_params",
                    "steps": "params:training.steps",
                    "learning_rate": "params:training.learning_rate",
                    "batch_size": "params:training.batch_size",
                    "log_entangling": "params:training.log_entangling",
                    "convergence_threshold": "params:training.convergence.threshold",
                    "convergence_steps": "params:training.convergence.steps",
                },
                outputs={
                    "model": "trained_model",
                    "params": "trained_params",
                    "grads": "trained_grads",
                    "coeffs": "trained_coefficients",
                },
                name="train_model",
            ),
        ]
    )
