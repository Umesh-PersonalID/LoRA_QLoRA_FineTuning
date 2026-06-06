import torch
from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)

from peft import (
    LoraConfig,
    get_peft_model,
)

from trl import SFTTrainer

MODEL_NAME = "google/gemma-3-1b-it"

train_dataset = load_dataset(
    "json",
    data_files="data/train_split.jsonl",
    split="train"
)

val_dataset = load_dataset(
    "json",
    data_files="data/val_split.jsonl",
    split="train"
)

def format_example(example):
    return {
        "text":
            f"<start_of_turn>user\n"
            f"{example['instruction']}"
            f"<end_of_turn>\n"
            f"<start_of_turn>model\n"
            f"{example['response']}"
            f"<end_of_turn>"
    }

train_dataset = train_dataset.map(format_example)
val_dataset = val_dataset.map(format_example)


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model.config.use_cache = False

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)

model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="outputs/lora_checkpoints",

    num_train_epochs=3,

    per_device_train_batch_size=1,
    per_device_eval_batch_size=2,

    gradient_accumulation_steps=16,

    learning_rate=2e-4,

    logging_steps=10,

    eval_strategy="steps",
    eval_steps=50,

    save_strategy="steps",
    save_steps=50,

    save_total_limit=2,

    bf16=True,
    fp16=False,

    optim="adamw_torch",

    lr_scheduler_type="cosine",
    warmup_ratio=0.05,

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",

    report_to="none",
)


trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    args=training_args,
)

trainer.train()

trainer.save_model("outputs/lora_adapter")
tokenizer.save_pretrained("outputs/lora_adapter")

print("LoRA Training Complete")
