import json
import random

random.seed(42)

with open("data/train.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

print(f"Total examples: {len(data)}")

# Shuffle
random.shuffle(data)

n = len(data)

train_end = int(0.80 * n)
val_end = int(0.90 * n)

train_data = data[:train_end]
val_data = data[train_end:val_end]
test_data = data[val_end:]

with open("data/train_split.jsonl", "w") as f:
    for item in train_data:
        f.write(json.dumps(item) + "\n")

with open("data/val_split.jsonl", "w") as f:
    for item in val_data:
        f.write(json.dumps(item) + "\n")

with open("data/test_split.jsonl", "w") as f:
    for item in test_data:
        f.write(json.dumps(item) + "\n")

print("\nDataset Split Complete")
print(f"Train: {len(train_data)}")
print(f"Validation: {len(val_data)}")
print(f"Test: {len(test_data)}")
