from kedro.pipeline import Pipeline, node, pipeline

from .nodes import train_model


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                func=train_model,
                inputs={
                    "model": "model",
                    "train_loader": "train_loader",
                    "noise_params": "params:model.noise_params",
                    "steps": "params:training.steps",
                    "learning_rate": "params:training.learning_rate",
                    "log_entangling": "params:training.log_entangling",
                    "log_coefficients": "params:training.log_coefficients",
                    "convergence_threshold": "params:training.convergence.threshold",
                    "convergence_gradient": "params:training.convergence.gradient",
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
