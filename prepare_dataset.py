import pandas as pd
import json

df = pd.read_csv("data/conversations.csv")

examples = []

for participant_id, group in df.groupby("ParticipantID"):

    group = group.sort_values("Utterance#")

    rows = group.to_dict("records")

    for i in range(len(rows)-1):

        current = rows[i]
        nxt = rows[i+1]

        if (
            current["Speaker"] == "client"
            and nxt["Speaker"] == "counsellor"
        ):

            prompt = str(current["Utterance"]).strip()
            response = str(nxt["Utterance"]).strip()

            if len(prompt) > 1 and len(response) > 1:

                examples.append({
                    "instruction": prompt,
                    "response": response
                })

print(f"Generated {len(examples)} examples")

with open("data/train.jsonl", "w") as f:
    for ex in examples:
        f.write(json.dumps(ex) + "\n")

print("Saved to data/train.jsonl")
