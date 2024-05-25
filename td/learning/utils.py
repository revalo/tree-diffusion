import lz4.frame
import numpy as np

from td.environments import Environment
from td.samplers import ConstrainedRandomSampler
from absl import logging


def binary_image_complexity(image: np.ndarray) -> float:
    """Get how complex a binary image is.

    Args:
        image (np.ndarray): Binary image.

    Returns:
        float: Complexity of the image between 0 and 1.
    """

    image_thresh = (image > 0.5).reshape(-1)
    data = image_thresh.tobytes()
    compressed = lz4.frame.compress(data)
    ratio = len(compressed) / len(data)

    return ratio


def rejection_sample_complex(
    env: Environment,
    observation: bool,
    threshold: float,
    sampler: ConstrainedRandomSampler,
    min_primitives: int = 1,
    max_primitives: int = 10,
    max_iter: int = 512,
) -> np.ndarray:
    attempts = 0
    while True:
        expression = sampler.sample(
            env.grammar.start_symbol, min_primitives, max_primitives
        )
        image = (
            env.compile(expression)
            if not observation
            else env.compile_observation(expression)
        )
        complexity = binary_image_complexity(image)
        if complexity >= threshold:
            break

        attempts += 1

        if attempts >= max_iter:
            logging.warning("Max attempts reached for complexity rejection sampling.")
            break

    return expression
