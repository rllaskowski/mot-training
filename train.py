import os
import re

import dotenv
import neptune
from datasets import load_dataset

import transformers
from transformers import (DataCollatorForLanguageModeling, GPT2Tokenizer,
                          MoTConfig, MoTLMHeadModel, Trainer,
                          TrainingArguments)

dotenv.load_dotenv()
transformers.logging.set_verbosity_info()


run = neptune.init_run(
    project=os.getenv("NEPTUNE_PROJECT"),
    api_token=os.getenv("NEPTUNE_API_TOKEN"),
)


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

config = MoTConfig(
    vocab_size=tokenizer.vocab_size,
    n_positions=1024,
    n_embd=256,
    n_layer=1,
    n_head=4,
    n_inner=1024,
    group_size=32,
)
model = MoTLMHeadModel(config)
model.train()

dataset = load_dataset("tiny_shakespeare")


tokenized_dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True), batched=True)
tokenizer.pad_token = tokenizer.eos_token

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)


training_args = TrainingArguments(
    output_dir="./gpt2_c4_trained",
    overwrite_output_dir=True,
    num_train_epochs=3,  # Adjust epochs as needed
    per_device_train_batch_size=4,  # Adjust batch size as needed
    save_steps=10_000,
    save_total_limit=2,
    report_to='neptune'
)


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    tokenizer=tokenizer,
)

trainer.train()

evaluate_results = trainer.evaluate()
run["epoch"].append(evaluate_results["epoch"])
