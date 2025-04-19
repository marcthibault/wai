import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GenerationConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# Model and tokenizer
model_id = "Qwen/Qwen2-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype=torch.bfloat16,  # Use bfloat16 on H100
    trust_remote_code=True,
)

tokenizer.pad_token = tokenizer.eos_token  # required for training padding

import json

# Dataset
from datasets import Dataset, Features, load_dataset

# Load the raw jsonl file
raw_dataset = load_dataset(
    "json",
    data_files="data/sa_tuples.txt",
)


# Transform each record to Chat-style `messages`
def convert_to_messages(example):
    instruction = example.get("instruction", "").strip()
    input_text = example.get("input", "").strip()
    output = example.get("output", "").strip()

    user_message = instruction
    if input_text:
        user_message += f"\n\n{input_text}"

    messages = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": output},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return {"text": text}


# Transform each record to Chat-style `messages`
def convert_to_messages_wai(example):
    input_text = repr(example.get("state", "")).strip()
    output = repr(example.get("action", "")).strip()

    user_message = input_text

    messages_train = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": output},
    ]
    text_train = tokenizer.apply_chat_template(messages_train, tokenize=False, add_generation_prompt=True)
    messages_test = [
        {"role": "user", "content": user_message},
    ]
    text_test = tokenizer.apply_chat_template(messages_test, tokenize=False, add_generation_prompt=True)
    return {
        "text_train": text_train,
        "text_test": text_test,
    }


# Map the dataset
chat_dataset = raw_dataset.map(convert_to_messages_wai)


# Tokenization
def tokenize_function(example):
    return tokenizer(
        example["text_train"],
        truncation=True,
        padding="max_length",
        max_length=1024,
    )


tokenized = chat_dataset.map(tokenize_function, batched=True)
tokenized = tokenized["train"].train_test_split(test_size=0.1)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Training args
training_args = TrainingArguments(
    output_dir="./qwen2-0.5b-finetuned",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="epoch",
    learning_rate=2e-5,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    fp16=False,  # H100 prefers bf16
    bf16=True,
    save_total_limit=2,
    report_to="none",  # or "wandb"
)


# Suggestion from https://wandb.ai/capecape/alpaca_ft/reports/How-to-Fine-tune-an-LLM-Part-3-The-HuggingFace-Trainer--Vmlldzo1OTEyNjMy#sampling-from-the-model-during-training
class LLMSampleCB(TrainerCallback):
    def __init__(self, trainer, tokenizer, test_dataset, num_samples=10, max_new_tokens=256, log_model="checkpoint"):
        "A CallBack to log samples a wandb.Table during training"
        self.sample_dataset = test_dataset.select(range(num_samples))
        self.model, self.tokenizer = trainer.model, tokenizer
        self.gen_config = GenerationConfig.from_pretrained(trainer.model.name_or_path, max_new_tokens=max_new_tokens)

    def generate(self, prompt):
        tokenized_prompt = self.tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
        with torch.inference_mode():
            output = self.model.generate(tokenized_prompt, generation_config=self.gen_config)
        return self.tokenizer.decode(output[0][len(tokenized_prompt[0]) :], skip_special_tokens=True)

    def samples_table(self, examples):
        "Create a wandb.Table to store the generations"
        records_table = []
        for example in tqdm(examples, leave=False):
            prompt = example["text_test"]
            generation = self.generate(prompt=prompt)
            records_table.append(
                {
                    "prompt": prompt,
                    "generation": generation,
                    **self.gen_config.to_dict(),
                }
            )
        return records_table

    def on_evaluate(self, args, state, control, **kwargs):
        "Log the wandb.Table after calling trainer.evaluate"
        records_table = self.samples_table(self.sample_dataset)
        print(records_table)


# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    data_collator=data_collator,
)

wandb_callback = LLMSampleCB(trainer, tokenizer, tokenized["train"], num_samples=10, max_new_tokens=256)
trainer.add_callback(wandb_callback)


# Train
trainer.train()


# TODO
# - Evaluate format accuracy
# - Fix EOS token management
