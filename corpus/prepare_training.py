"""Prepare training data for Qwen3-0.6B ClojureScript fine-tuning.

Splits training_pairs.jsonl into train/valid sets,
filters overly long samples, and writes train.jsonl + valid.jsonl.
"""
import json
import random

random.seed(42)

# Load and filter
pairs = []
skipped = 0
with open("corpus/training_pairs.jsonl") as f:
    for line in f:
        d = json.loads(line)
        total_chars = sum(len(m["content"]) for m in d["messages"])
        if total_chars > 2048:
            skipped += 1
            continue
        pairs.append(d)

print(f"Loaded {len(pairs)} pairs ({skipped} skipped as >2048 chars)")

# Shuffle and split 95/5
random.shuffle(pairs)
split = int(len(pairs) * 0.95)
train = pairs[:split]
valid = pairs[split:]

print(f"Train: {len(train)}, Valid: {len(valid)}")

# Write
with open("corpus/train.jsonl", "w") as f:
    for d in train:
        f.write(json.dumps(d) + "\n")

with open("corpus/valid.jsonl", "w") as f:
    for d in valid:
        f.write(json.dumps(d) + "\n")

print("Wrote corpus/train.jsonl and corpus/valid.jsonl")
