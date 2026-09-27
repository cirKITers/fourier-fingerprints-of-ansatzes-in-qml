from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    expressibility,
)


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                func=expressibility,
                inputs={
                    "model": "model",
                    "samples": "params:expressibility.n_samples",
                    "seed": "params:seed",
                    "n_bins": "params:expressibility.n_bins",
                    "input_domain": "params:data.fourier.domain",
                    "noise_params": "params:model.noise_params",
                },
                outputs="expressibility",
                name="expressibility",
            ),
        ]
    )
