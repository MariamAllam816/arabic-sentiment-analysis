"""
train.py
Fine-tune AraBERT on an Arabic sentiment dataset.

Usage:
    python src/train.py \
        --data_path data/astd_processed.csv \
        --model_name aubmindlab/bert-base-arabertv2 \
        --epochs 4 \
        --batch_size 16 \
        --output_dir models/arabert-sentiment
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)
import evaluate

from preprocess import ArabicPreprocessor, LABEL2ID, ID2LABEL


# ─── Reproducibility ─────────────────────────────────────────────────────────

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_data(data_path: str, preprocessor: ArabicPreprocessor) -> DatasetDict:
    """
    Load CSV with columns: text, label  (label = Positive/Negative/Neutral)
    Applies preprocessing, stratified split (80/10/10), returns HuggingFace DatasetDict.
    """
    df = pd.read_csv(data_path)
    assert 'text' in df.columns and 'label' in df.columns, \
        "CSV must have 'text' and 'label' columns"

    df = df.dropna(subset=['text', 'label'])
    df['text'] = preprocessor.clean_batch(df['text'].tolist())
    df = df[df['text'].str.len() > 3].reset_index(drop=True)
    df['label'] = df['label'].map(LABEL2ID)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df['label'], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42
    )

    print(f"  Train : {len(train_df):,} samples")
    print(f"  Val   : {len(val_df):,} samples")
    print(f"  Test  : {len(test_df):,} samples")

    return DatasetDict({
        'train': Dataset.from_pandas(train_df[['text', 'label']].reset_index(drop=True)),
        'validation': Dataset.from_pandas(val_df[['text', 'label']].reset_index(drop=True)),
        'test': Dataset.from_pandas(test_df[['text', 'label']].reset_index(drop=True)),
    })


# ─── Tokenization ─────────────────────────────────────────────────────────────

def make_tokenize_fn(tokenizer, max_len: int):
    def tokenize(batch):
        return tokenizer(
            batch['text'],
            truncation=True,
            max_length=max_len,
            padding=False,          # DataCollatorWithPadding handles dynamic padding
        )
    return tokenize


# ─── Metrics ──────────────────────────────────────────────────────────────────

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average='macro')
    return {**acc, **f1}


# ─── Main training loop ───────────────────────────────────────────────────────

def train(args):
    seed_everything(42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print(f"  Arabic Sentiment — AraBERT Fine-tuning")
    print(f"  Device : {device.upper()}")
    print(f"  Model  : {args.model_name}")
    print(f"{'='*60}\n")

    # 1. Preprocessing
    preprocessor = ArabicPreprocessor()

    # 2. Load data
    print("[1/5] Loading and preprocessing data...")
    dataset = load_data(args.data_path, preprocessor)

    # 3. Tokenizer
    print("[2/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    tokenize_fn = make_tokenize_fn(tokenizer, args.max_len)
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=['text'])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 4. Model
    print("[3/5] Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # 5. Training arguments
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type='linear',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        logging_dir=str(output_dir / 'logs'),
        logging_steps=50,
        fp16=torch.cuda.is_available(),       # AMP on GPU
        dataloader_num_workers=2,
        report_to='none',                     # set to 'wandb' if you want W&B tracking
        seed=42,
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized['train'],
        eval_dataset=tokenized['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 7. Train
    print("[4/5] Training...\n")
    trainer.train()

    # 8. Save
    print(f"\n[5/5] Saving best model to {output_dir}...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # 9. Final test-set evaluation
    print("\n── Test Set Evaluation ──────────────────────────────────")
    results = trainer.evaluate(tokenized['test'])
    for k, v in results.items():
        print(f"  {k:<30} {v:.4f}")
    print("─" * 50)
    print("\n✅ Training complete!")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune AraBERT for Arabic sentiment')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to CSV with text,label columns')
    parser.add_argument('--model_name', type=str,
                        default='aubmindlab/bert-base-arabertv2',
                        help='HuggingFace model name or local path')
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--output_dir', type=str,
                        default='models/arabert-sentiment')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
