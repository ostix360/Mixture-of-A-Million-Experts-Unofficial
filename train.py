import torch
from datasets import load_dataset

from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling

from config import MixtralConfig
from model import MixtralForCausalLM

config = MixtralConfig(
    vocab_size=32000,
    hidden_size=256,
    intermediate_size=0,
    num_hidden_layers=6,
    num_attention_heads=8,
    num_key_value_heads=1,
    hidden_act="silu",
    num_experts_per_tok=26,
    num_local_experts=30,
    output_router_logits=False,
    moe_heads=8,
    moe_k_dim=64,
    query_batchnorm=True,
)

model = MixtralForCausalLM(config)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer.model_max_length = 512
tokenizer.pad_token = tokenizer.eos_token

train_dataset = load_dataset("wikipedia", "20220301.simple", split="train[:20000]", trust_remote_code=True)
eval_dataset = load_dataset("wikipedia", "20220301.simple", split="train[20000:20200]", trust_remote_code=True)


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
    )


train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=1,
    learning_rate=5e-4,
    per_device_train_batch_size=7,
    per_device_eval_batch_size=4,
    logging_steps=50,
    report_to=["none"],
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    bf16_full_eval=torch.cuda.is_bf16_supported(),
    fp16_full_eval=not torch.cuda.is_bf16_supported(),
    optim="adamw_8bit",
    lr_scheduler_type="linear",
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=512,
)

print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

trainer.train()
