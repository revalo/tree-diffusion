import os
import pickle
import random
import uuid

import numpy as np
import torch
import torch.nn.functional as F
from absl import app, flags, logging
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

import wandb
from td.environments import Environment, environments
from td.learning.gpt import TransformerConfig, TreeDiffusion, ValueHead
from td.learning.tokenizer import Tokenizer
from td.samplers import ConstrainedRandomSampler
from td.samplers.mutator import forward_process_with_path

FLAGS = flags.FLAGS
flags.DEFINE_integer("batch_size", 256, "Batch size")
flags.DEFINE_integer("print_every", 100, "Print every")
flags.DEFINE_integer("training_steps", -1, "Training steps")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "Checkpoint directory")
flags.DEFINE_integer("checkpoint_steps", 10000, "Checkpoint steps")
flags.DEFINE_integer("num_workers", 16, "Number of workers for data loading")
flags.DEFINE_integer("max_steps", 5, "Minimum number of steps")
flags.DEFINE_float("learning_rate", 3e-4, "Learning rate")
flags.DEFINE_bool("wandb", True, "Log to wandb")
flags.DEFINE_string("env", "rainbow", "Environment to use")
flags.DEFINE_integer("max_sequence_length", 512, "Maximum sequence length")
flags.DEFINE_integer("min_primitives", 2, "Minimum number of primitives")
flags.DEFINE_integer("max_primitives", 8, "Maximum number of primitives")
flags.DEFINE_integer("n_layers", 3, "Number of layers")
flags.DEFINE_string("device", "cuda", "Device to use")
flags.DEFINE_string("resume_from", None, "Resume from checkpoint")
flags.DEFINE_bool(
    "target_observation", False, "Use observation compiler for target image."
)
flags.DEFINE_bool(
    "current_observation", False, "Use observation compiler for current image."
)
flags.DEFINE_string("base_model_path", None, "Base model path")


class ValueNetDataset(IterableDataset):
    def __init__(
        self,
        batch_size,
        env_name,
        min_steps,
        max_steps,
        min_primitives,
        max_primitives,
        target_observation,
        current_observation,
    ):
        self._env_name = env_name
        self._batch_size = batch_size
        self._min_steps = min_steps
        self._max_steps = max_steps
        self._min_primitives = min_primitives
        self._max_primitives = max_primitives
        self._target_observation = target_observation
        self._current_observation = current_observation

    def _produce_batch(self):
        def sample_fn():
            return self._sampler.sample(
                self._env.grammar.start_symbol,
                min_primitives=self._min_primitives,
                max_primitives=self._max_primitives,
            )

        def sample_batch_element():
            target_expression = self._env.sample_non_empty(sample_fn)
            steps = random.randint(self._min_steps, self._max_steps)
            mutated_expression, _, path = forward_process_with_path(
                target_expression,
                steps,
                self._env.grammar,
                self._sampler,
                min_primitives=self._min_primitives,
                max_primitives=self._max_primitives,
                return_full_path=True,
            )

            target_image = (
                self._env.compile(target_expression)
                if not self._target_observation
                else self._env.compile_observation(target_expression)
            )
            mutated_image = (
                self._env.compile(mutated_expression)
                if not self._current_observation
                else self._env.compile_observation(mutated_expression)
            )

            return target_image, mutated_image, len(path)

        batch = []
        while len(batch) < self._batch_size:
            try:
                batch.append(sample_batch_element())
            except Exception as e:
                logging.error(f"Error while sampling batch element: {e}")
                pass

        target_images, mutated_images, path_lengths = zip(*batch)

        return (
            np.array(target_images).transpose(0, 3, 1, 2),
            np.array(mutated_images).transpose(0, 3, 1, 2),
            np.array(path_lengths),
        )

    def __iter__(self):
        worker_info = get_worker_info()

        if worker_info is not None:
            np.random.seed(worker_info.id)
            random.seed(worker_info.id)

        self._env: Environment = environments[self._env_name]()
        self._sampler = ConstrainedRandomSampler(self._env.grammar)

        while True:
            yield self._produce_batch()


def loss_fn(main_model: TreeDiffusion, value_head: ValueHead, batch):
    target_images, mutated_images, path_lengths = batch
    image_embeddings = main_model.image_embeddings(target_images, mutated_images)
    image_embeddings = image_embeddings.detach().squeeze(1)

    predictions = value_head(image_embeddings).squeeze(-1)

    # MSE loss.
    loss = F.mse_loss(predictions, path_lengths.float())

    # MAE metric.
    mae = torch.mean(torch.abs(predictions - path_lengths.float()))

    return loss, (mae,)


def generate_uuid():
    return str(uuid.uuid4())


def batch_to_torch(batch, device="cpu"):
    target_images, mutated_images, path_lengths = batch

    return (
        target_images.to(device).float(),
        mutated_images.to(device).float(),
        path_lengths.to(device).long(),
    )


def main(argv):
    env = environments[FLAGS.env]()
    tokenizer = Tokenizer(
        env.grammar,
        max_token_length=FLAGS.max_sequence_length,
        max_sequence_length=FLAGS.max_sequence_length,
    )
    random.seed(1)

    local_run_id = FLAGS.resume_from or generate_uuid()
    checkpoint_dir = (
        os.path.join(FLAGS.checkpoint_dir, local_run_id)
        if FLAGS.checkpoint_dir
        else None
    )
    step = 0

    config = {
        "notes": "value-head",
        "batch_size": FLAGS.batch_size,
        "learning_rate": FLAGS.learning_rate,
        "env": FLAGS.env,
        "max_steps": FLAGS.max_steps,
        "local_run_id": local_run_id,
        "max_sequence_length": FLAGS.max_sequence_length,
        "max_primitives": FLAGS.max_primitives,
        "min_primitives": FLAGS.min_primitives,
        "n_layers": FLAGS.n_layers,
        "target_observation": FLAGS.target_observation,
        "current_observation": FLAGS.current_observation,
        "base_model_path": FLAGS.base_model_path,
    }

    if FLAGS.wandb:
        wandb.init(
            project="tree-diffusion",
            config=config,
        )

    if FLAGS.base_model_path is None or not os.path.exists(FLAGS.base_model_path):
        raise ValueError(f"Invalid base model path: {FLAGS.base_model_path}")

    with open(FLAGS.base_model_path, "rb") as f:
        base_model_state = pickle.load(f)

    base_model_config = base_model_state["config"]
    base_model_state = base_model_state["model"]

    assert (
        base_model_config["env"] == FLAGS.env
    ), "Environment mismatch between base model and current model"

    base_model = TreeDiffusion(
        TransformerConfig(
            vocab_size=tokenizer.vocabulary_size,
            max_seq_len=tokenizer.max_sequence_length,
            n_layer=base_model_config["n_layers"],
            n_head=base_model_config["num_heads"],
            n_embd=base_model_config["d_model"],
        ),
        input_channels=env.compiled_shape[-1],
        image_model_name=base_model_config["image_model"],
    ).to(FLAGS.device)
    base_model.load_state_dict(base_model_state)

    value_head = ValueHead(
        n_embd=base_model_config["d_model"],
        n_layers=FLAGS.n_layers,
    ).to(FLAGS.device)

    if os.path.exists(checkpoint_dir):
        checkpoint_files = [
            f
            for f in os.listdir(checkpoint_dir)
            if f.endswith(".pt") and f.startswith(f"{FLAGS.env}_step_")
        ]
        if checkpoint_files:
            latest_checkpoint = max(
                checkpoint_files, key=lambda x: int(x.split("_")[-1].split(".")[0])
            )
            checkpoint_filename = os.path.join(checkpoint_dir, latest_checkpoint)
            with open(checkpoint_filename, "rb") as f:
                value_head.load_state_dict(pickle.load(f))
            step = int(latest_checkpoint.split("_")[-1].split(".")[0])
            logging.info(
                f"Loaded checkpoint from {checkpoint_filename}, starting at step {step}"
            )

    optimizer = torch.optim.Adam(value_head.parameters(), lr=FLAGS.learning_rate)

    logging.info("Starting to train!")

    if (
        checkpoint_dir
        and FLAGS.checkpoint_steps > 0
        and not os.path.exists(checkpoint_dir)
    ):
        logging.info(
            f"Local run ID: {local_run_id}, saving checkpoints to {checkpoint_dir}"
        )
        os.makedirs(checkpoint_dir)

    batch_metrics = []

    batch_size = FLAGS.batch_size
    env_name = FLAGS.env
    max_steps = FLAGS.max_steps
    min_primitives = FLAGS.min_primitives
    max_primitives = FLAGS.max_primitives

    dataset = ValueNetDataset(
        batch_size,
        env_name,
        1,
        max_steps,
        min_primitives,
        max_primitives,
        FLAGS.target_observation,
        FLAGS.current_observation,
    )

    dataloader = DataLoader(dataset, batch_size=None, num_workers=FLAGS.num_workers)

    value_head.train()

    for batch in dataloader:
        batch = batch_to_torch(batch, FLAGS.device)
        loss, aux = loss_fn(base_model, value_head, batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mae = aux[0]

        batch_metrics.append((loss, mae))

        if step % FLAGS.print_every == FLAGS.print_every - 1:
            metrics = np.mean(
                np.array(torch.tensor(batch_metrics).detach().cpu()), axis=0
            )
            mean_loss, mean_mae = metrics
            logging.info(f"Step {step + 1}, Loss: {mean_loss:.4f}, MAE: {mean_mae:.2f}")

            if FLAGS.wandb:
                wandb.log(
                    {
                        "loss": mean_loss,
                        "mae": mean_mae,
                    },
                    step=step + 1,
                )

            batch_metrics.clear()

        if (
            checkpoint_dir
            and FLAGS.checkpoint_steps > 0
            and step % FLAGS.checkpoint_steps == FLAGS.checkpoint_steps - 1
        ):
            checkpoint_filename = os.path.join(
                checkpoint_dir, f"{env_name}_step_{step + 1}.pt"
            )
            if os.path.exists(checkpoint_filename):
                logging.warning(
                    f"Checkpoint file {checkpoint_filename} already exists, skipping."
                )
            else:
                with open(checkpoint_filename, "wb") as f:
                    pickle.dump({"model": value_head.state_dict(), "config": config}, f)
                logging.info(f"Checkpointed state to {checkpoint_filename}")

        step += 1

        if FLAGS.training_steps > 0 and step >= FLAGS.training_steps:
            break


if __name__ == "__main__":
    app.run(main)
