import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from bert_score import score
from rouge_score import rouge_scorer

BASE_MODEL = "google/gemma-3-1b-it"
ADAPTER_PATH = "outputs/final_adapter"

test_data = []

with open("data/test_split.jsonl", "r") as f:
    for line in f:
        test_data.append(json.loads(line))

print(f"Loaded {len(test_data)} test examples")


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

ft_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

ft_model = PeftModel.from_pretrained(
    ft_model,
    ADAPTER_PATH
)


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
            temperature=0.7,
            do_sample=False
        )

    result = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return result.split("model")[-1].strip()

references = []

base_outputs = []

ft_outputs = []

for sample in tqdm(test_data):

    prompt = sample["instruction"]
    target = sample["response"]

    references.append(target)

    base_pred = generate(base_model, prompt)
    ft_pred = generate(ft_model, prompt)

    base_outputs.append(base_pred)
    ft_outputs.append(ft_pred)

print("\nComputing BERTScore...\n")

P_base, R_base, F1_base = score(
    base_outputs,
    references,
    lang="en",
    verbose=True
)

P_ft, R_ft, F1_ft = score(
    ft_outputs,
    references,
    lang="en",
    verbose=True
)

print("\n========== BERTScore ==========\n")

print(
    f"Base Model F1: {F1_base.mean():.4f}"
)

print(
    f"Fine-Tuned F1: {F1_ft.mean():.4f}"
)


scorer = rouge_scorer.RougeScorer(
    ["rougeL"],
    use_stemmer=True
)

base_rouge = []
ft_rouge = []

for ref, b, f in zip(
    references,
    base_outputs,
    ft_outputs
):

    base_rouge.append(
        scorer.score(ref, b)["rougeL"].fmeasure
    )

    ft_rouge.append(
        scorer.score(ref, f)["rougeL"].fmeasure
    )

print("\n========== ROUGE-L ==========\n")

print(
    f"Base ROUGE-L: {sum(base_rouge)/len(base_rouge):.4f}"
)

print(
    f"Fine-Tuned ROUGE-L: {sum(ft_rouge)/len(ft_rouge):.4f}"
)

print("\n========== Samples ==========\n")

for i in range(5):

    print(f"\nPrompt: {test_data[i]['instruction']}")

    print(f"\nReference:")
    print(references[i])

    print(f"\nBase:")
    print(base_outputs[i])

    print(f"\nFine-Tuned:")
    print(ft_outputs[i])

    print("\n" + "="*80)
