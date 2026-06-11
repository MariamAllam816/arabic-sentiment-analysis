"""
evaluate.py
Full evaluation of a fine-tuned AraBERT sentiment model.
Produces: accuracy, macro-F1, per-class report, and confusion matrix plot.

Usage:
    python src/evaluate.py \
        --model_path models/arabert-sentiment \
        --test_path data/test.csv
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from preprocess import ArabicPreprocessor, LABEL2ID, ID2LABEL


# ─── Inference on test CSV ────────────────────────────────────────────────────

def predict_from_csv(model_path: str, test_path: str, batch_size: int = 32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    preprocessor = ArabicPreprocessor()
    df = pd.read_csv(test_path).dropna(subset=['text', 'label'])
    df['text'] = preprocessor.clean_batch(df['text'].tolist())
    df = df[df['text'].str.len() > 3].reset_index(drop=True)
    df['label_id'] = df['label'].map(LABEL2ID)
    df = df.dropna(subset=['label_id'])

    texts = df['text'].tolist()
    true_ids = df['label_id'].astype(int).tolist()

    all_preds = []
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i: i + batch_size]
        enc = tokenizer(
            batch_texts,
            truncation=True,
            max_length=128,
            padding=True,
            return_tensors='pt',
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=-1)

        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())

    return true_ids, all_preds, all_probs


# ─── Report helpers ───────────────────────────────────────────────────────────

def print_report(true_ids, pred_ids):
    labels = list(LABEL2ID.keys())
    label_order = [LABEL2ID[l] for l in labels]

    print("\n── Classification Report ────────────────────────────────")
    print(classification_report(
        true_ids, pred_ids,
        labels=label_order,
        target_names=labels,
        digits=4,
    ))


def plot_confusion_matrix(true_ids, pred_ids, save_path: str = None):
    labels = list(LABEL2ID.keys())
    label_order = [LABEL2ID[l] for l in labels]

    cm = confusion_matrix(true_ids, pred_ids, labels=label_order)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm_pct,
        annot=True,
        fmt='.2%',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Arabic Sentiment — Confusion Matrix', fontsize=13, pad=12)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Confusion matrix saved → {save_path}")
    else:
        plt.show()


# ─── Confidence histogram ─────────────────────────────────────────────────────

def plot_confidence_histogram(all_probs, pred_ids, save_path: str = None):
    confidences = [probs[pred] for probs, pred in zip(all_probs, pred_ids)]

    plt.figure(figsize=(7, 4))
    plt.hist(confidences, bins=30, color='steelblue', edgecolor='white', alpha=0.85)
    plt.xlabel('Prediction Confidence', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Confidence Distribution', fontsize=13)
    plt.axvline(np.mean(confidences), color='red', linestyle='--',
                label=f'Mean: {np.mean(confidences):.2f}')
    plt.legend()
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Confidence histogram saved → {save_path}")
    else:
        plt.show()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    true_ids, pred_ids, all_probs = predict_from_csv(
        args.model_path, args.test_path, batch_size=args.batch_size
    )

    print_report(true_ids, pred_ids)

    if args.save_plots:
        plot_confusion_matrix(true_ids, pred_ids,
                              save_path='results/confusion_matrix.png')
        plot_confidence_histogram(all_probs, pred_ids,
                                  save_path='results/confidence_hist.png')
    else:
        plot_confusion_matrix(true_ids, pred_ids)
        plot_confidence_histogram(all_probs, pred_ids)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Arabic sentiment model')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--test_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--save_plots', action='store_true',
                        help='Save plots to results/ instead of showing interactively')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
