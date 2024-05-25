from absl import app
from absl import flags
from absl import logging

from td.environments import Environment, environments
from td.learning.tokenizer import Tokenizer
from td.learning.gpt import TreeDiffusion, TransformerConfig
from td.samplers import ConstrainedRandomSampler
from td.learning.constrained_decoding import ar_decoder, sample_model_kv

import pickle
import numpy as np
import os
import uuid
import wandb
import torch

flags.DEFINE_string("checkpoint_name", None, "Path to the checkpoint to evaluate")
flags.DEFINE_string("ar_checkpoint_name", None, "Path to the AR checkpoint.")
flags.DEFINE_string("problem_filename", None, "Number of problems to evaluate")
flags.DEFINE_integer("max_steps", 100, "Maximum number of steps to take")
flags.DEFINE_integer("evaluation_batch_size", 16, "Batch size for evaluation")
flags.DEFINE_integer("num_replicas", 32, "Batch size for evaluation")
flags.DEFINE_float("temperature", 1.0, "Temperature for sampling")
flags.DEFINE_string("evaluation_dir", "evals", "Evaluations directory")
flags.DEFINE_bool("wandb", True, "Log to wandb")
flags.DEFINE_string("device", "cuda", "Device to use")

FLAGS = flags.FLAGS


def generate_uuid():
    return str(uuid.uuid4())


def load_model(checkpoint_name, device):
    with open(checkpoint_name, "rb") as f:
        state = pickle.load(f)

    config = state["config"]

    env_name = config["env"]
    image_model = config["image_model"]
    d_model = config["d_model"]
    n_layers = config["n_layers"]
    num_heads = config["num_heads"]
    max_sequence_length = config["max_sequence_length"]
    target_observation = config["target_observation"]

    for key, value in config.items():
        logging.info(f"{key}: {value}")

    env: Environment = environments[env_name]()
    sampler = ConstrainedRandomSampler(env.grammar)
    tokenizer = Tokenizer(
        env.grammar,
        max_token_length=max_sequence_length,
        max_sequence_length=max_sequence_length,
    )

    model = TreeDiffusion(
        TransformerConfig(
            vocab_size=tokenizer.vocabulary_size,
            max_seq_len=tokenizer.max_sequence_length,
            n_layer=n_layers,
            n_head=num_heads,
            n_embd=d_model,
        ),
        input_channels=env.compiled_shape[-1],
        image_model_name=image_model,
    )

    model.load_state_dict(state["model"])
    model.to(device)

    return model, env, tokenizer, sampler, target_observation, config


def main(argv):
    logging.info(f"Evaluating {FLAGS.checkpoint_name}")

    if not os.path.exists(FLAGS.evaluation_dir):
        os.makedirs(FLAGS.evaluation_dir)

    local_run_id = generate_uuid()
    logging.info(f"Local run id: {local_run_id}")

    save_filename = os.path.join(FLAGS.evaluation_dir, f"{local_run_id}.pkl")

    td_model, env, tokenizer, sampler, target_observation, _ = load_model(
        FLAGS.checkpoint_name, FLAGS.device
    )
    ar_model, _, ar_tokenizer, _, ar_to, ar_config = load_model(
        FLAGS.ar_checkpoint_name, FLAGS.device
    )

    config = {
        "notes": "td-eval",
        "temperature": FLAGS.temperature,
        "max_steps": FLAGS.max_steps,
        "evaluation_batch_size": FLAGS.evaluation_batch_size,
        "checkpoint_name": FLAGS.checkpoint_name,
        "local_run_id": local_run_id,
        "ar_checkpoint_name": FLAGS.ar_checkpoint_name,
        "num_replicas": FLAGS.num_replicas,
    }

    if FLAGS.wandb:
        wandb.init(
            project="tree-diffusion",
            config=config,
        )

    with open(FLAGS.problem_filename, "rb") as f:
        target_expressions = pickle.load(f)

    target_images = np.array(
        [
            env.compile(e) if not target_observation else env.compile_observation(e)
            for e in target_expressions
        ]
    )

    target_images_torch = (
        torch.tensor(target_images).to(FLAGS.device).float().permute(0, 3, 1, 2)
    )

    steps_to_solve = np.zeros(len(target_expressions)) + np.inf

    for problem_i in range(len(target_expressions)):
        logging.info(f"Problem {problem_i + 1} / {len(target_expressions)} ...")

        target_image_torch = target_images_torch[problem_i].unsqueeze(0)
        # Replicate the target image to create a batch.
        batch_targets = target_image_torch.repeat(FLAGS.num_replicas, 1, 1, 1)

        ar_predictions = ar_decoder(
            ar_model,
            env,
            ar_tokenizer,
            ar_config["num_image_tokens"],
            batch_targets,
            temperature=1.0,
        )

        initial_expressions = list(set(ar_predictions))[: FLAGS.num_replicas]
        logging.info(f"Unique AR predictions: {len(initial_expressions)}")
        while len(initial_expressions) < FLAGS.num_replicas:
            initial_expressions.append(sampler.sample(env.grammar.start_symbol))

        current_expressions = [x for x in initial_expressions]
        current_images = np.array([env.compile(e) for e in current_expressions])

        # Did we already solve the problem?
        for image_i in range(len(current_images)):
            if env.goal_reached(current_images[image_i], target_images[problem_i]):
                steps_to_solve[problem_i] = image_i + 1
                break

        if steps_to_solve[problem_i] < np.inf:
            logging.info(f"Steps to solve: {steps_to_solve[problem_i]}")
            current_solve_rate = np.sum(steps_to_solve < np.inf) / (problem_i + 1)
            logging.info(f"Solve rate: {current_solve_rate * 100:.2f}%")
            with open(save_filename, "wb") as f:
                pickle.dump(
                    {
                        "steps_to_solve": steps_to_solve,
                    },
                    f,
                )
            continue

        # We've already spent replicas.
        current_steps = len(current_images)

        values = [-np.inf]
        for step_i in range(FLAGS.max_steps):
            print(f"Step {step_i} / {FLAGS.max_steps} ... {max(values)}")
            mutations = sample_model_kv(
                td_model,
                env,
                tokenizer,
                current_expressions,
                batch_targets,
                temperature=FLAGS.temperature,
            )

            current_expressions = [
                m.apply(e) for m, e in zip(mutations, current_expressions)
            ]
            current_images = np.array([env.compile(e) for e in current_expressions])

            for image_i in range(len(current_images)):
                if env.goal_reached(current_images[image_i], target_images[problem_i]):
                    steps_to_solve[problem_i] = current_steps + image_i + 1
                    break
                values.append(
                    env._goal_checker.goal_reached_value(
                        current_images[image_i], target_images[problem_i]
                    )
                )

            current_steps += len(current_images)

            if steps_to_solve[problem_i] < np.inf:
                break

        logging.info(f"Max val: {max(values)}")
        logging.info(f"Steps to solve: {steps_to_solve[problem_i]}")
        current_solve_rate = np.sum(steps_to_solve < np.inf) / (problem_i + 1)
        logging.info(f"Solve rate: {current_solve_rate * 100:.2f}%")
        with open(save_filename, "wb") as f:
            pickle.dump(
                {
                    "steps_to_solve": steps_to_solve,
                },
                f,
            )


if __name__ == "__main__":
    app.run(main)
