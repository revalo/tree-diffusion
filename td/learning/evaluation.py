import random
from typing import List, NamedTuple

import iceberg as ice
import numpy as np
import torch
import tqdm
from absl import logging
from scipy.optimize import linear_sum_assignment

from td.environments import Environment
from td.learning.constrained_decoding import sample_model_kv, ar_decoder
from td.learning.tokenizer import Tokenizer
from td.samplers import ConstrainedRandomSampler
from td.samplers.mutator import random_mutation


class OneStepEvaluator(object):
    class EvaluationResult(NamedTuple):
        goal_reached: float
        error_rate: float

    def __init__(
        self,
        env: Environment,
        sampler: ConstrainedRandomSampler,
        tokenizer: Tokenizer,
        seed: int = 0,
        num_problems: int = 256,
        evaluation_batch_size: int = 8,
        min_primitives: int = 2,
        max_primitives: int = 7,
        device: str = "cpu",
        target_observation: bool = False,
        non_empty: bool = True,
    ):
        self._env = env
        self._tokenizer = tokenizer
        self._sampler = sampler
        self._num_problems = num_problems
        self._evaluation_batch_size = evaluation_batch_size
        self._target_observation = target_observation
        self._non_empty = non_empty

        random.seed(seed)

        def sample_fn():
            return sampler.sample(
                env.grammar.start_symbol,
                min_primitives=min_primitives,
                max_primitives=max_primitives,
            )

        # Generate 1-step problems.
        self._target_expressions = [
            env.sample_non_empty(sample_fn) if non_empty else sample_fn()
            for _ in range(num_problems)
        ]

        self._target_images = np.array(
            [env.compile(e) for e in self._target_expressions]
        )
        self._target_images_observation = (
            np.array([env.compile_observation(e) for e in self._target_expressions])
            if target_observation
            else self._target_images
        )
        self._target_images_observation_torch = (
            torch.tensor(self._target_images_observation)
            .float()
            .permute(0, 3, 1, 2)
            .to(device)
        )
        self._mutations = [
            random_mutation(e, env.grammar, sampler) for e in self._target_expressions
        ]
        self._current_expressions = [
            m.apply(e) for e, m in zip(self._target_expressions, self._mutations)
        ]

    def evaluate(
        self, model, progress_bar: bool = False, sample_func=sample_model_kv
    ) -> EvaluationResult:
        try:
            predicted_reverse_mutations = []

            for i in tqdm.trange(
                0,
                self._num_problems,
                self._evaluation_batch_size,
                disable=not progress_bar,
            ):
                batch_mutated_expressions = self._current_expressions[
                    i : i + self._evaluation_batch_size
                ]
                batch_target_images = self._target_images_observation_torch[
                    i : i + self._evaluation_batch_size
                ]

                batch_predicted_reverse_mutations = sample_func(
                    model,
                    self._env,
                    self._tokenizer,
                    batch_mutated_expressions,
                    batch_target_images,
                    temperature=0.1,
                )

                predicted_reverse_mutations.extend(batch_predicted_reverse_mutations)

            total_goal_reached = 0
            error_count = 0

            for (
                target_image,
                mutated_expression,
                predicted_reverse_mutation,
            ) in zip(
                self._target_images,
                self._current_expressions,
                predicted_reverse_mutations,
            ):
                try:
                    recovered_target_image = self._env.compile(
                        predicted_reverse_mutation.apply(mutated_expression)
                    )
                except Exception as _:
                    error_count += 1
                    continue

                if self._env.goal_reached(target_image, recovered_target_image):
                    total_goal_reached += 1

            goal_reached = total_goal_reached / self._num_problems
            error_rate = error_count / self._num_problems

            return OneStepEvaluator.EvaluationResult(goal_reached, error_rate)
        except Exception as e:
            logging.error(f"Error during evaluation: {e}")
            return OneStepEvaluator.EvaluationResult(0.0, 1.0)


class AREvaluator(object):
    class EvaluationResult(NamedTuple):
        goal_reached: float
        error_rate: float
        goal_reached_indexes: np.ndarray
        predicted_expressions: List[str]

    def __init__(
        self,
        env: Environment,
        sampler: ConstrainedRandomSampler,
        tokenizer: Tokenizer,
        num_image_tokens: int,
        seed: int = 0,
        num_problems: int = 256,
        evaluation_batch_size: int = 8,
        min_primitives: int = 2,
        max_primitives: int = 7,
        device: str = "cpu",
        target_observation: bool = False,
        non_empty: bool = True,
        temperature: float = 0.4,
        target_expressions: List[str] = None,
    ):
        self._env = env
        self._tokenizer = tokenizer
        self._num_image_tokens = num_image_tokens
        self._sampler = sampler
        self._num_problems = num_problems
        self._evaluation_batch_size = evaluation_batch_size
        self._target_observation = target_observation
        self._non_empty = non_empty
        self._temperature = temperature

        random.seed(seed)

        def sample_fn():
            return sampler.sample(
                env.grammar.start_symbol,
                min_primitives=min_primitives,
                max_primitives=max_primitives,
            )

        # Generate problems.
        if target_expressions is not None:
            self._target_expressions = target_expressions
        else:
            self._target_expressions = [
                env.sample_non_empty(sample_fn) if non_empty else sample_fn()
                for _ in range(num_problems)
            ]

        self._target_images = np.array(
            [env.compile(e) for e in self._target_expressions]
        )
        self._target_images_observation = (
            np.array([env.compile_observation(e) for e in self._target_expressions])
            if target_observation
            else self._target_images
        )
        self._target_images_observation_torch = (
            torch.tensor(self._target_images_observation)
            .float()
            .permute(0, 3, 1, 2)
            .to(device)
        )

    def evaluate(
        self,
        model,
        progress_bar: bool = False,
    ) -> EvaluationResult:
        try:
            predicted_expressions = []

            for i in tqdm.trange(
                0,
                self._num_problems,
                self._evaluation_batch_size,
                disable=not progress_bar,
            ):
                batch_target_images = self._target_images_observation_torch[
                    i : i + self._evaluation_batch_size
                ]

                batch_predicted_reverse_mutations = ar_decoder(
                    model,
                    self._env,
                    self._tokenizer,
                    self._num_image_tokens,
                    batch_target_images,
                    temperature=self._temperature,
                )

                predicted_expressions.extend(batch_predicted_reverse_mutations)

            total_goal_reached = 0
            error_count = 0

            goal_reached_indexes = np.zeros(self._num_problems, dtype=bool)

            for i, (
                target_image,
                predicted_expression,
            ) in enumerate(
                zip(
                    self._target_images,
                    predicted_expressions,
                )
            ):
                try:
                    recovered_target_image = self._env.compile(predicted_expression)
                except Exception as _:
                    error_count += 1
                    continue

                if self._env.goal_reached(target_image, recovered_target_image):
                    total_goal_reached += 1
                    goal_reached_indexes[i] = True

            goal_reached = total_goal_reached / self._num_problems
            error_rate = error_count / self._num_problems

            return AREvaluator.EvaluationResult(
                goal_reached, error_rate, goal_reached_indexes, predicted_expressions
            )
        except Exception as e:
            logging.error(f"Error during evaluation: {e}")
            return AREvaluator.EvaluationResult(
                0.0,
                1.0,
                np.zeros(self._num_problems, dtype=bool),
                [],
            )



def _scene_leaves_and_centers(scene: ice.Drawable):
    leaves = scene.find_all(lambda d: len(d.children) == 0)
    leaf_bounds = [scene.child_bounds(leaf).center for leaf in leaves]

    return leaves, np.array(leaf_bounds)


def _get_scene(env: Environment, expression: str):
    return env.compiler._expression_to_iceberg.transform(env.grammar.parse(expression))


def _expression_subset(expression: str, ice_objects: List[ice.Drawable]):
    return [
        expression[o._lark_meta.start_pos : o._lark_meta.end_pos] for o in ice_objects
    ]


def semantic_scene_distance(
    env: Environment, tokenizer: Tokenizer, expressionA: str, expressionB: str
) -> int:
    """Get the number of "tokens" that are different between two scenes. This will only work with very specialized environments.
    Specifically, we need the environment to have a compiler that can transform an expression into an iceberg scene. And each iceberg object must have
    a `_lark_meta` attribute that contains the start and end position of the object in the original expression. This is used to extract the object from the expression.

    Args:
        env (Environment): The environment that contains the compiler and grammar.
        tokenizer (Tokenizer): The tokenizer used to tokenize the expressions.
        expressionA (str): The first expression.
        expressionB (str): The second expression.

    Returns:
        int: The number of tokens that are different between the two scenes.
    """

    sceneA = _get_scene(env, expressionA)
    sceneB = _get_scene(env, expressionB)

    sceneA_objects, sceneA_centers = _scene_leaves_and_centers(sceneA)
    sceneB_objects, sceneB_centers = _scene_leaves_and_centers(sceneB)

    sceneA_objects = _expression_subset(expressionA, sceneA_objects)
    sceneB_objects = _expression_subset(expressionB, sceneB_objects)

    cost_matrix = np.zeros((len(sceneA_objects), len(sceneB_objects)))
    for i, centerA in enumerate(sceneA_centers):
        for j, centerB in enumerate(sceneB_centers):
            cost_matrix[i, j] = ((centerA - centerB) ** 2).sum()

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    total_cost = 0

    unmatched_objects = []
    # Find number of objects that were not matched.
    for i in range(len(sceneA_objects)):
        if i not in row_ind:
            unmatched_objects.append(sceneA_objects[i])

    for j in range(len(sceneB_objects)):
        if j not in col_ind:
            unmatched_objects.append(sceneB_objects[j])

    total_cost += sum(len(tokenizer._tokenize_one(o)) for o in unmatched_objects)

    for i, j in zip(row_ind, col_ind):
        tokenized_i = tokenizer._tokenize_one(sceneA_objects[i])
        tokenized_j = tokenizer._tokenize_one(sceneB_objects[j])

        # How many tokens are different between the two objects. Their length is not necessarily the same.
        total_cost += abs(len(tokenized_i) - len(tokenized_j))

        # Count number of tokens.
        trim_length = min(len(tokenized_i), len(tokenized_j))

        total_cost += np.sum(tokenized_i[:trim_length] != tokenized_j[:trim_length])

    return total_cost
