import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
)
from trl import SFTTrainer

MODEL_NAME = "google/gemma-3-1b-it"
 
train_dataset = load_dataset("json", data_files="data/train_split.jsonl", split="train")
val_dataset   = load_dataset("json", data_files="data/val_split.jsonl",   split="train")
 
def format_example(example):
    return {
        "text":
            f"<start_of_turn>user\n{example['instruction']}<end_of_turn>\n"
            f"<start_of_turn>model\n{example['response']}<end_of_turn>"
    }
 
train_dataset = train_dataset.map(format_example)
val_dataset   = val_dataset.map(format_example)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,   # fix: was float16
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="outputs/checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    bf16=True,
    fp16=False,
    optim="paged_adamw_8bit",
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

trainer.save_model("outputs/final_adapter")
tokenizer.save_pretrained("outputs/final_adapter")
print("Training Complete")

