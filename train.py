import os

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
            "n_embd": 256,
            "n_layer": 4,
            "n_head": 4,
            "n_inner": 1024,
            "group_size": 32,
        },
        "training_args": {
            "overwrite_output_dir": True,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "save_steps": 10_000,
            "save_total_limit": 2,
        },
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


def main():
    _setup()
    _init_neptune_run()

    experiment = EXPERIMENTS[EXPERIMENT_NAME]

    config = MoTConfig(**experiment["mot_config"])

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", vocab_size=config.vocab_size)
    tokenizer.pad_token = tokenizer.eos_token

    model = MoTLMHeadModel(config)
    model.train()

    # Fetching the smallest variant of C4. Available variants are:
    # - en: 305GB in JSON format
    # - en.noblocklist: 380GB in JSON format
    # - en.noclean: 2.3TB in JSON format
    # - realnewslike: 15GB in JSON format
    dataset = load_dataset("c4", "realnewslike")

    tokenized_dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True), batched=True)

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
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
    )

    trainer.train()

    trainer.save_model()


if __name__ == "__main__":
    main()
