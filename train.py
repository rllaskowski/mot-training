import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from sched import scheduler
import torch
import dotenv
import neptune
from datasets import load_dataset
from transformers.optimization import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import transformers
from transformers import DataCollatorForLanguageModeling
from transformers import GPT2TokenizerFast as GPT2Tokenizer
from transformers import MoTConfig, MoTLMHeadModel, Trainer, TrainingArguments
from transformers.integrations import NeptuneCallback

EXPERIMENTS = {
    "mot_medium_32_8": {
        # Original MoT-Medium/32E/8 model config from the paper
        # https://arxiv.org/pdf/2310.15961.pdf
        "mot_config": {
            "vocab_size": 50257,
            "n_positions": 256,
            "expert_size": 256,
            "n_expert": 256,
            "n_embd": 512,
            "n_layer": 8,
            # n_inner is ignored if expert_size is not None, but should not be None
            # TODO: fix in the MoT implementation so that it is not required
            "n_inner": 32,
            "n_head": 8,
            "group_size": 32,
        },
        "training_args": {
            "learning_rate": 7e-4,
            "per_device_train_batch_size": 256,
            "per_device_eval_batch_size": 256,
            "overwrite_output_dir": True,
            "max_steps": 150_000,
            "save_steps": 25_000,
            "fp16": True,
            "save_total_limit": 2,
            "logging_steps": 100,
        },
        "dataset": ["c4", "en"],
        "tokenizer": {
            "config": "gpt2",
            "args": {
                "vocab_size": 50257,
                "batched": True,
                "max_length": 256,
            },
        },
    },
    "really_small_c4": {
        "mot_config": {
            "vocab_size": 50257,
            "n_positions": 1024,
            "expert_size": 256,
            "n_expert": 256,
            "n_embd": 64,
            "n_layer": 1,
            "n_head": 2,
            "n_inner": 128,
            "group_size": 4,
        },
        "training_args": {
            "learning_rate": 7e-4,
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 8,
            "overwrite_output_dir": True,
            "max_steps": 150,
            "save_steps": 150,
            "save_total_limit": 2,
        },
        "dataset": ["c4", "realnewslike"],
        "tokenizer": {
            "config": "gpt2",
            "args": {
                "vocab_size": 50257,
                "batched": True,
                "max_length": 256,
            },
        },
    },
}

BASE_DIR = Path(__file__).resolve().parent


@dataclass
class Args:

    experiment_name: str
    torch_verbosity_info: bool

    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument("--experiment-name", type=str, required=True)
        parser.add_argument("--torch-verbosity-info", action="store_true", default=True)
        return cls(**vars(parser.parse_args()))

    @property
    def experiment_dir(self):
        return BASE_DIR / "experiments" / self.experiment_name

    def __post_init__(self):
        if self.experiment_name not in EXPERIMENTS:
            raise ValueError(f"Unknown experiment name: {self.experiment_name}")


def _init_neptune_run():
    return neptune.init_run(
        project=os.getenv("NEPTUNE_PROJECT"),
        api_token=os.getenv("NEPTUNE_API_TOKEN"),
    )


def _setup() -> Args:
    args = Args.from_args()

    dotenv.load_dotenv()
    if args.torch_verbosity_info:
        transformers.logging.set_verbosity_info()

    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)

    return args


def _get_optimizer(args: Args, model):
    experiment = EXPERIMENTS[args.experiment_name]
    opt = torch.optim.AdamW(model.parameters(), lr=experiment["training_args"]["learning_rate"])

    scheduler = get_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps=2500,
        num_training_steps=experiment["training_args"]["max_steps"],
        num_cycles=0.467,  # so that it ends at around 10% of the original learning rate
    )

    return opt, scheduler


def _get_dataset(tokenizer, args: Args):
    raw_dataset = load_dataset(*EXPERIMENTS[args.experiment_name]["dataset"], streaming=True)

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenized_dataset = raw_dataset.map(tokenize_fn, batched=True)

    return tokenized_dataset


def _get_tokenizer(args: Args):
    tokenizer = GPT2Tokenizer.from_pretrained(
        EXPERIMENTS[args.experiment_name]["tokenizer"]["config"],
        **EXPERIMENTS[args.experiment_name]["tokenizer"]["args"],
    )
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def main():
    args = _setup()

    experiment = EXPERIMENTS[args.experiment_name]

    run = _init_neptune_run()
    tokenizer = _get_tokenizer(args)
    dataset = _get_dataset(tokenizer, args)

    config = MoTConfig(**experiment["mot_config"])
    model = MoTLMHeadModel(config)

    print("Model parameters:", model.num_parameters() / 1e6, "M")

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.experiment_dir,
        logging_dir=args.experiment_dir / "logs",
        report_to=["neptune"],
        **experiment["training_args"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        callbacks=[NeptuneCallback(run=run)],
        optimizers=_get_optimizer(args, model),
    )

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
