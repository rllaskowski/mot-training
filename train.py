import os
from re import split

import dotenv
import neptune
from datasets import load_dataset

import transformers
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2Tokenizer,
    MoTConfig,
    MoTLMHeadModel,
    Trainer,
    TrainingArguments,
)
from pathlib import Path


EXPERIMENTS = {
    # Original MoT model config from the paper
    # https://arxiv.org/pdf/2310.15961.pdf
    "mot_paper_small_c4": {
        "mot_config": {
            "vocab_size": 50257,
            "n_positions": 1024,
            "2048"
            "n_embd": 256,
            "n_layer": 8,
            "n_head": 4,
            "n_inner": 1024,
            "group_size": 32,
        },
        "training_args": {
            "learning_rate": 7e-4,
            "per_device_train_batch_size": 256,
            "per_device_eval_batch_size": 256,
            "overwrite_output_dir": True,
            "max_steps": 150_000,
            "save_steps": 10_000,
            "save_total_limit": 2,
        },
        "dataset": ["c4", "realnewslike"],
        "tokenizer": {
            "from_pretrained": "gpt2",
            "args": {
                "vocab_size": 50257,
                "batched": True,
                "max_length": 256,
            }
        }
    }
}

TORCH_VERBOSITY_INFO = False
BASE_DIR = Path(__file__).resolve().parent

EXPERIMENT_NAME = "mot_paper_small_c4"

EXPERIMENT_DIR = BASE_DIR / "experiments" / EXPERIMENT_NAME


def _init_neptune_run():
    return neptune.init_run(
        project=os.getenv("NEPTUNE_PROJECT"),
        api_token=os.getenv("NEPTUNE_API_TOKEN"),
    )


def _setup():
    dotenv.load_dotenv()
    if TORCH_VERBOSITY_INFO:
        transformers.logging.set_verbosity_info()

    if not EXPERIMENT_DIR.exists():
        EXPERIMENT_DIR.mkdir(parents=True)


def _get_dataset(tokenizer):
    # Fetching the smallest variant of C4. Available variants are:
    # - en: 305GB in JSON format
    # - en.noblocklist: 380GB in JSON format
    # - en.noclean: 2.3TB in JSON format
    # - realnewslike: 15GB in JSON format
    dataset_dir = '/local_storage_1/rllaskowski/'
    dataset = load_dataset(**EXPERIMENTS[EXPERIMENT_NAME]["dataset"], data_dir=dataset_dir)
    return dataset.map(lambda x: tokenizer(x["text"], truncation=True), batched=True)


def _get_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained(
        EXPERIMENTS[EXPERIMENT_NAME]["tokenizer"]["from_pretrained"],
        **EXPERIMENTS[EXPERIMENT_NAME]["tokenizer"]["args"],
    )

    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def main():
    _setup()
    _init_neptune_run()

    experiment = EXPERIMENTS[EXPERIMENT_NAME]

    tokenizer = _get_tokenizer()
    dataset = _get_dataset(tokenizer)

    config = MoTConfig(**experiment["mot_config"])
    model = MoTLMHeadModel(config)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=EXPERIMENT_DIR,
        logging_dir=EXPERIMENT_DIR / "logs",
        **experiment["training_args"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
    )

    trainer.train()

    trainer.save_model()


if __name__ == "__main__":
    main()
