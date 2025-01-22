"""Project pipelines."""

from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from saqml.pipelines.data_generation.pipeline import (
    create_pipeline as create_data_generation_pipeline,
)
from saqml.pipelines.data_science.coefficients.pipeline import (
    create_pipeline as create_coefficients_pipeline,
)
from saqml.pipelines.data_science.coefficients.pipeline import (
    create_randcoeffs_pipeline as create_randcoeffs_pipeline,
)
from saqml.pipelines.data_science.expressibility.pipeline import (
    create_pipeline as create_expressibility_pipeline,
)
from saqml.pipelines.data_science.training.pipeline import (
    create_pipeline as create_training_pipeline,
)

from saqml.pipelines.visualization.pipeline import (
    create_pipeline as create_visualization_pipeline,
)

from saqml.pipelines.visualization.pipeline import (
    create_model_pipeline as create_model_visualization_pipeline,
)

from saqml.pipelines.visualization.pipeline import (
    create_randcoeffs_pipeline as create_randcoeffs_viz_pipeline,
)


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = {
        "__default__": create_data_generation_pipeline()
        + create_coefficients_pipeline()
        + create_training_pipeline()
        + create_visualization_pipeline()
        + create_model_visualization_pipeline(),
        "training": create_data_generation_pipeline()
        + create_training_pipeline()
        + create_model_visualization_pipeline(),    
        "coefficients": create_data_generation_pipeline()
        + create_coefficients_pipeline()
        + create_visualization_pipeline(),
        "randcoeffs": create_data_generation_pipeline()
        + create_randcoeffs_pipeline()
        + create_randcoeffs_viz_pipeline(),
        "expressibility": create_data_generation_pipeline()
        + create_expressibility_pipeline(),
        "coeffexpr": create_data_generation_pipeline()
        + create_coefficients_pipeline()
        + create_expressibility_pipeline()
        + create_visualization_pipeline(),
    }
    return pipelines
