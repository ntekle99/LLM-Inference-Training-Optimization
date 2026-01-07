# finetuning.py — from‑scratch PyTorch loop using custom LLaMA classes + auto‑download 200 Alpaca samples

import os
import json
import argparse
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

# your custom llama imports
from llama.tokenizer import Tokenizer
from llama.model import ModelArgs, Llama

torch.manual_seed(1)

ALPACA_RAW_URL = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"


def ensure_alpaca_jsonl(path: Path, limit: int):
    """
    If `path` doesn’t exist, download the full alpaca_data.json
    and write the first `limit` entries as a .jsonl file.
    """
    if path.exists():
        return
    print(f"→ Downloading Alpaca data from {ALPACA_RAW_URL} …")
    resp = urllib.request.urlopen(ALPACA_RAW_URL)
    data = json.loads(resp.read().decode("utf-8"))
    subset = data[:limit]
    print(f"→ Writing {len(subset)} samples to {path}")
    with open(path, "w") as out:
        for ex in subset:
            out.write(json.dumps(ex) + "\n")


class AlpacaDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                obj = json.loads(line)
                instr = obj["instruction"].strip()
                inp = obj.get("input", "").strip()
                out = obj["output"].strip()
                if inp:
                    prompt = (
                        f"### Instruction:\n{instr}\n\n"
                        f"### Input:\n{inp}\n\n"
                        f"### Response:\n{out}"
                    )
                else:
                    prompt = (
                        f"### Instruction:\n{instr}\n\n"
                        f"### Response:\n{out}"
                    )
                self.samples.append(prompt)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        # supply required bos and eos flags
        ids = self.tokenizer.encode(text, bos=True, eos=True)
        # truncate to max_length
        ids = ids[: self.max_length]
        return torch.tensor(ids, dtype=torch.long)


def collate_fn(batch):
    max_len = max(b.size(0) for b in batch)
    input_ids, attention_mask, labels = [], [], []

    for ids in batch:
        pad_len = max_len - ids.size(0)
        input_ids.append(
            torch.cat([ids, torch.full((pad_len,), tokenizer.pad_id, dtype=torch.long)])
        )
        attention_mask.append(
            torch.cat([torch.ones(ids.size(0), dtype=torch.long),
                       torch.zeros(pad_len, dtype=torch.long)])
        )
        # For causal LM, labels shift input_ids, with pad masked to -100
        labels.append(
            torch.cat([ids[1:], torch.full((pad_len+1,), -100, dtype=torch.long)])
        )

    return {
        "input_ids":      torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels":         torch.stack(labels)
    }


def get_linear_scheduler(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - step) /
            float(max(1, num_training_steps - num_warmup_steps))
        )
    return LambdaLR(optimizer, lr_lambda)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", type=Path, required=True,
                   help="path to ~/.llama/checkpoints/Llama3.2-1B")
    p.add_argument("--data-file", type=Path, default=Path("alpaca_200.jsonl"),
                   help="where to write/read the 200-sample Alpaca JSONL")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--accumulation-steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--limit", type=int, default=200,
                   help="number of Alpaca samples to slice")
    p.add_argument("--fp16", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.data_file.parent.mkdir(parents=True, exist_ok=True)

    # 1) Download + slice data
    ensure_alpaca_jsonl(args.data_file, args.limit)

    # 2) Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3) Load tokenizer + model
    ckpt = args.checkpoint_dir.expanduser()
    tok_path = ckpt / "tokenizer.model"
    model_path = ckpt / "consolidated.00.pth"

    tokenizer = Tokenizer(str(tok_path))

    # load checkpoint directly onto device
    checkpoint = torch.load(str(model_path), map_location=device)
    model_args = ModelArgs()
    model = Llama(model_args)
    model.load_state_dict(checkpoint, strict=True)
    if args.fp16:
        model.half()
    model.to(device)
    model.train()

    # 4) Data loader
    ds = AlpacaDataset(str(args.data_file), tokenizer, args.max_length)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    # 5) Optimizer + scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-6)
    total_steps = len(loader) // args.accumulation_steps * args.epochs
    scheduler = get_linear_scheduler(optimizer, args.warmup_steps, total_steps)

    # 6) Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # 7) Training loop
    global_step = 0
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        optimizer.zero_grad()
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch.get("labels")
            if labels is not None:
                labels = labels.to(device)

            with torch.cuda.amp.autocast(enabled=args.fp16):
                logits = model(input_ids, attention_mask=attention_mask)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                # mask padding
                mask = attention_mask[..., 1:]
                shift_labels = shift_labels.masked_fill(mask == 0, -100)
                loss = loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                ) / args.accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.accumulation-steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                pbar.set_postfix({"loss": f"{(loss.item()*args.accumulation-steps):.3f}",
                                  "lr": scheduler.get_last_lr()[0]})

    # 8) Save finetuned weights
    out_path = args.output_dir / "finetuned.pth"
    torch.save(model.state_dict(), str(out_path))
    print(f"✅ Saved finetuned checkpoint to {out_path}")
