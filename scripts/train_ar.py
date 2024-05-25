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
from td.learning.evaluation import AREvaluator
from td.learning.gpt import TransformerConfig, TreeDiffusion
from td.learning.tokenizer import Tokenizer
from td.samplers import ConstrainedRandomSampler

FLAGS = flags.FLAGS
flags.DEFINE_integer("batch_size", 256, "Batch size")
flags.DEFINE_integer("print_every", 100, "Print every")
flags.DEFINE_integer("test_every", 10000, "Test every")
flags.DEFINE_integer("training_steps", -1, "Training steps")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "Checkpoint directory")
flags.DEFINE_integer("checkpoint_steps", 10000, "Checkpoint steps")
flags.DEFINE_integer("num_workers", 16, "Number of workers for data loading")
flags.DEFINE_float("learning_rate", 3e-4, "Learning rate")
flags.DEFINE_bool("wandb", True, "Log to wandb")
flags.DEFINE_string("env", "rainbow", "Environment to use")
flags.DEFINE_integer("num_test_expressions", 256, "Number of test expressions")
flags.DEFINE_integer("max_sequence_length", 512, "Maximum sequence length")
flags.DEFINE_integer("min_primitives", 2, "Minimum number of primitives")
flags.DEFINE_integer("max_primitives", 8, "Maximum number of primitives")
flags.DEFINE_integer("n_layers", 3, "Number of layers")
flags.DEFINE_integer("d_model", 128, "Model dimension")
flags.DEFINE_integer("num_heads", 8, "Number of heads")
flags.DEFINE_string("device", "cuda", "Device to use")
flags.DEFINE_string("image_model", "nf_resnet26", "Vision model to use")
flags.DEFINE_string("resume_from", None, "Resume from checkpoint")
flags.DEFINE_bool(
    "target_observation", False, "Use observation compiler for target image."
)
flags.DEFINE_integer("num_image_tokens", 4, "Number of image tokens")


class ARDataset(IterableDataset):
    def __init__(
        self,
        batch_size,
        env_name,
        max_sequence_length,
        min_primitives,
        max_primitives,
        target_observation,
        num_image_tokens,
    ):
        self._env_name = env_name
        self._batch_size = batch_size
        self._max_sequence_length = max_sequence_length
        self._min_primitives = min_primitives
        self._max_primitives = max_primitives
        self._target_observation = target_observation
        self._num_image_tokens = num_image_tokens

    def _produce_batch(self):
        def sample_fn():
            return self._sampler.sample(
                self._env.grammar.start_symbol,
                min_primitives=self._min_primitives,
                max_primitives=self._max_primitives,
            )

        def sample_batch_element():
            target_expression = self._env.sample_non_empty(sample_fn)
            if self._target_observation:
                target_image = self._env.compile_observation(target_expression)
            else:
                target_image = self._env.compile(target_expression)
            return target_image, target_expression

        batch = []
        while len(batch) < self._batch_size:
            try:
                batch.append(sample_batch_element())
            except Exception as e:
                logging.error(f"Error while sampling batch element: {e}")
                pass

        tokenized = []
        context_tokens_mask = []
        target_images, target_expressions = zip(*batch)

        for target_expression in target_expressions:
            target_expression_tokens = self._tokenizer._tokenize_one(target_expression)

            tokens = (
                [self._tokenizer.pad_token] * self._num_image_tokens
                + [self._tokenizer.sos_token]
                + target_expression_tokens
                + [self._tokenizer.eos_token]
            )

            if len(tokens) > self._tokenizer.max_sequence_length:
                logging.warning(
                    f"Token sequence too long {len(tokens)} > {self._tokenizer.max_sequence_length}. Skipping batch."
                )
                tokenized.append(None)
                context_tokens_mask.append(None)
                break

            tokenized.append(
                tokens
                + [self._tokenizer.pad_token]
                * (self._tokenizer.max_sequence_length - len(tokens))
            )
            context_tokens_mask.append(
                [0] * self._num_image_tokens
                + [1] * (len(target_expression_tokens) + 1)
                + [0]
                * (
                    self._tokenizer.max_sequence_length
                    - self._num_image_tokens
                    - len(target_expression_tokens)
                    - 1
                )
            )

        if any(t is None for t in tokenized):
            return self._produce_batch()

        return (
            np.array(tokenized),
            np.array(context_tokens_mask),
            np.array(target_images).transpose(0, 3, 1, 2),
        )

    def __iter__(self):
        worker_info = get_worker_info()

        if worker_info is not None:
            np.random.seed(worker_info.id)
            random.seed(worker_info.id)

        self._env: Environment = environments[self._env_name]()
        self._sampler = ConstrainedRandomSampler(self._env.grammar)
        self._tokenizer = Tokenizer(
            self._env.grammar,
            max_token_length=self._max_sequence_length,
            max_sequence_length=self._max_sequence_length,
        )

        while True:
            yield self._produce_batch()


def loss_fn(model, batch):
    tokens, mask, target_images = batch
    logits = model(tokens, target_images, target_images * 0.0)

    logits = logits[:, :-1]
    targets = tokens[:, 1:]
    mask = mask[:, 1:]

    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
    )
    loss = loss.reshape(targets.shape)
    loss = (loss * mask).sum() / mask.sum()

    return loss


def generate_uuid():
    return str(uuid.uuid4())


def batch_to_torch(batch, device="cpu"):
    tokens, mask, target_images = batch

    return (
        tokens.to(device).long(),
        mask.to(device).float(),
        target_images.to(device).float(),
    )


def main(argv):
    env = environments[FLAGS.env]()
    sampler = ConstrainedRandomSampler(env.grammar)
    tokenizer = Tokenizer(
        env.grammar,
        max_token_length=FLAGS.max_sequence_length,
        max_sequence_length=FLAGS.max_sequence_length,
    )
    ar_evaluator = AREvaluator(
        env,
        sampler,
        tokenizer,
        FLAGS.num_image_tokens,
        num_problems=FLAGS.num_test_expressions,
        device=FLAGS.device,
        evaluation_batch_size=FLAGS.batch_size,
        target_observation=FLAGS.target_observation,
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
        "notes": "AR-baseline",
        "batch_size": FLAGS.batch_size,
        "learning_rate": FLAGS.learning_rate,
        "env": FLAGS.env,
        "local_run_id": local_run_id,
        "max_sequence_length": FLAGS.max_sequence_length,
        "max_primitives": FLAGS.max_primitives,
        "min_primitives": FLAGS.min_primitives,
        "n_layers": FLAGS.n_layers,
        "d_model": FLAGS.d_model,
        "num_heads": FLAGS.num_heads,
        "image_model": FLAGS.image_model,
        "target_observation": FLAGS.target_observation,
        "num_image_tokens": FLAGS.num_image_tokens,
    }

    if FLAGS.wandb:
        wandb.init(
            project="tree-diffusion",
            config=config,
        )

    model = TreeDiffusion(
        TransformerConfig(
            vocab_size=tokenizer.vocabulary_size,
            max_seq_len=tokenizer.max_sequence_length,
            n_layer=FLAGS.n_layers,
            n_head=FLAGS.num_heads,
            n_embd=FLAGS.d_model,
        ),
        input_channels=env.compiled_shape[-1],
        image_model_name=FLAGS.image_model,
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
                model.load_state_dict(pickle.load(f))
            step = int(latest_checkpoint.split("_")[-1].split(".")[0])
            logging.info(
                f"Loaded checkpoint from {checkpoint_filename}, starting at step {step}"
            )

    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate)

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
    max_sequence_length = FLAGS.max_sequence_length
    min_primitives = FLAGS.min_primitives
    max_primitives = FLAGS.max_primitives

    dataset = ARDataset(
        batch_size=batch_size,
        env_name=env_name,
        max_sequence_length=max_sequence_length,
        min_primitives=min_primitives,
        max_primitives=max_primitives,
        target_observation=FLAGS.target_observation,
        num_image_tokens=FLAGS.num_image_tokens,
    )

    dataloader = DataLoader(dataset, batch_size=None, num_workers=FLAGS.num_workers)

    model.train()

    for batch in dataloader:
        batch = batch_to_torch(batch, FLAGS.device)
        loss = loss_fn(model, batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_metrics.append((loss,))

        if step % FLAGS.print_every == FLAGS.print_every - 1:
            metrics = np.mean(
                np.array(torch.tensor(batch_metrics).detach().cpu()), axis=0
            )
            mean_loss = metrics[0].item()
            logging.info(f"Step {step + 1}, Loss: {mean_loss:.4f}")

            if FLAGS.wandb:
                wandb.log(
                    {
                        "loss": mean_loss,
                    },
                    step=step + 1,
                )

            batch_metrics.clear()

        if FLAGS.test_every > 0 and step % FLAGS.test_every == FLAGS.test_every - 1:
            eval_result = ar_evaluator.evaluate(model, progress_bar=False)

            logging.info(
                f"Step {step + 1}, Goal reached: {eval_result.goal_reached:.2f}, Error rate: {eval_result.error_rate:.2f}"
            )

            if FLAGS.wandb:
                wandb.log(
                    {
                        "goal_reached": eval_result.goal_reached,
                        "error_rate": eval_result.error_rate,
                    },
                    step=step + 1,
                )

            model.train()

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
                    pickle.dump({"model": model.state_dict(), "config": config}, f)
                logging.info(f"Checkpointed state to {checkpoint_filename}")

        step += 1

        if FLAGS.training_steps > 0 and step >= FLAGS.training_steps:
            break


if __name__ == "__main__":
    app.run(main)
