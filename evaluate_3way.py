import json
import torch

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from bert_score import score
from rouge_score import rouge_scorer

BASE_MODEL = "google/gemma-3-1b-it"
QLORA_PATH = "outputs/final_adapter"
LORA_PATH = "outputs/lora_adapter"

# ==================================================
# Load Test Set
# ==================================================

test_data = []

with open("data/test_split.jsonl", "r") as f:
    for line in f:
        test_data.append(json.loads(line))

print(f"Loaded {len(test_data)} test examples")

# ==================================================
# Tokenizer
# ==================================================

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# ==================================================
# Base Model
# ==================================================

print("\nLoading Base Model...")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

# ==================================================
# QLoRA Model
# ==================================================

print("Loading QLoRA Model...")

qlora_base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

qlora_model = PeftModel.from_pretrained(
    qlora_base,
    QLORA_PATH
)

# ==================================================
# LoRA Model
# ==================================================

print("Loading LoRA Model...")

lora_base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

lora_model = PeftModel.from_pretrained(
    lora_base,
    LORA_PATH
)

# ==================================================
# Generation Function
# ==================================================

def generate(model, prompt):

    text = (
        f"<start_of_turn>user\n"
        f"{prompt}"
        f"<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False,
            temperature=0.7
        )

    result = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    if "model" in result:
        result = result.split("model")[-1]

    return result.strip()

# ==================================================
# Generate Predictions
# ==================================================

references = []

base_outputs = []
qlora_outputs = []
lora_outputs = []

print("\nGenerating predictions...\n")

for sample in tqdm(test_data):

    prompt = sample["instruction"]
    reference = sample["response"]

    references.append(reference)

    base_outputs.append(
        generate(base_model, prompt)
    )

    qlora_outputs.append(
        generate(qlora_model, prompt)
    )

    lora_outputs.append(
        generate(lora_model, prompt)
    )

# ==================================================
# BERTScore
# ==================================================

print("\nComputing BERTScore...\n")

_, _, base_f1 = score(
    base_outputs,
    references,
    lang="en",
    verbose=True
)

_, _, qlora_f1 = score(
    qlora_outputs,
    references,
    lang="en",
    verbose=True
)

_, _, lora_f1 = score(
    lora_outputs,
    references,
    lang="en",
    verbose=True
)

# ==================================================
# ROUGE-L
# ==================================================

scorer = rouge_scorer.RougeScorer(
    ["rougeL"],
    use_stemmer=True
)

base_rouge = []
qlora_rouge = []
lora_rouge = []

for ref, b, q, l in zip(
    references,
    base_outputs,
    qlora_outputs,
    lora_outputs
):

    base_rouge.append(
        scorer.score(ref, b)["rougeL"].fmeasure
    )

    qlora_rouge.append(
        scorer.score(ref, q)["rougeL"].fmeasure
    )

    lora_rouge.append(
        scorer.score(ref, l)["rougeL"].fmeasure
    )

# ==================================================
# Results
# ==================================================

print("\n")
print("=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(
    f"Base Model BERTScore F1 : {base_f1.mean():.4f}"
)

print(
    f"QLoRA BERTScore F1      : {qlora_f1.mean():.4f}"
)

print(
    f"LoRA BERTScore F1       : {lora_f1.mean():.4f}"
)

print()

print(
    f"Base Model ROUGE-L      : {sum(base_rouge)/len(base_rouge):.4f}"
)

print(
    f"QLoRA ROUGE-L           : {sum(qlora_rouge)/len(qlora_rouge):.4f}"
)

print(
    f"LoRA ROUGE-L            : {sum(lora_rouge)/len(lora_rouge):.4f}"
)

# ==================================================
# Samples
# ==================================================

print("\n")
print("=" * 60)
print("SAMPLE OUTPUTS")
print("=" * 60)

for i in range(5):

    print("\nPrompt:")
    print(test_data[i]["instruction"])

    print("\nReference:")
    print(references[i])

    print("\nBase:")
    print(base_outputs[i])

    print("\nQLoRA:")
    print(qlora_outputs[i])

    print("\nLoRA:")
    print(lora_outputs[i])

    print("\n" + "=" * 80)
