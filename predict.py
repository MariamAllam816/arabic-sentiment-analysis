"""
predict.py
Run Arabic sentiment inference on single texts or batches.
Can be used as a CLI tool or imported as a module.

Usage (CLI):
    python src/predict.py --model_path models/arabert-sentiment --text "المنتج ممتاز جداً"
    python src/predict.py --model_path models/arabert-sentiment --file data/new_reviews.txt

Usage (module):
    from src.predict import ArabicSentimentClassifier
    clf = ArabicSentimentClassifier("models/arabert-sentiment")
    print(clf.predict("المنتج ممتاز جداً"))
"""

import argparse
from pathlib import Path
from typing import Union

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from preprocess import ArabicPreprocessor, ID2LABEL


# ─── Classifier class ─────────────────────────────────────────────────────────

class ArabicSentimentClassifier:
    """
    Wraps a fine-tuned AraBERT model for Arabic sentiment prediction.

    Args:
        model_path: Path to the saved model directory (output of train.py)
        device: 'cuda', 'cpu', or 'auto' (default).
    """

    LABEL_EMOJI = {
        'Positive': '✅',
        'Negative': '❌',
        'Neutral':  '⚖️',
    }

    def __init__(self, model_path: str, device: str = 'auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        self.preprocessor = ArabicPreprocessor()

    def predict(self, text: str, return_all_scores: bool = False) -> dict:
        """
        Predict sentiment for a single Arabic text.

        Returns:
            {
              'label': 'Positive',
              'confidence': 0.942,
              'scores': {'Positive': 0.942, 'Neutral': 0.041, 'Negative': 0.017}  # if return_all_scores
            }
        """
        cleaned = self.preprocessor.clean(text)
        if not cleaned:
            return {'label': 'Neutral', 'confidence': 0.0, 'raw_text': text}

        enc = self.tokenizer(
            cleaned,
            truncation=True,
            max_length=128,
            return_tensors='pt',
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

        label_id = int(probs.argmax())
        label = ID2LABEL[label_id]
        confidence = float(probs[label_id])

        result = {
            'label': label,
            'confidence': round(confidence, 4),
            'emoji': self.LABEL_EMOJI[label],
        }

        if return_all_scores:
            result['scores'] = {
                ID2LABEL[i]: round(float(p), 4) for i, p in enumerate(probs)
            }

        return result

    def predict_batch(self, texts: list[str], batch_size: int = 32) -> list[dict]:
        """Predict sentiment for a list of texts."""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            cleaned = [self.preprocessor.clean(t) for t in batch]

            enc = self.tokenizer(
                cleaned,
                truncation=True,
                max_length=128,
                padding=True,
                return_tensors='pt',
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            with torch.no_grad():
                logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            for j, p in enumerate(probs):
                label_id = int(p.argmax())
                label = ID2LABEL[label_id]
                results.append({
                    'text': batch[j],
                    'label': label,
                    'confidence': round(float(p[label_id]), 4),
                    'emoji': self.LABEL_EMOJI[label],
                })
        return results

    def pretty_print(self, text: str):
        """Predict and print a nicely formatted result."""
        result = self.predict(text, return_all_scores=True)
        print(f"\n  Input : {text}")
        print(f"  Result: {result['emoji']} {result['label']}  "
              f"(confidence: {result['confidence']*100:.1f}%)")
        if 'scores' in result:
            for lbl, score in sorted(result['scores'].items(),
                                     key=lambda x: -x[1]):
                bar = '█' * int(score * 20)
                print(f"         {lbl:<10} {bar:<20} {score*100:.1f}%")
        print()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Arabic Sentiment Inference')
    parser.add_argument('--model_path', type=str,
                        default='models/arabert-sentiment')
    parser.add_argument('--text', type=str, default=None,
                        help='Single Arabic text to classify')
    parser.add_argument('--file', type=str, default=None,
                        help='Path to .txt file with one text per line')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    clf = ArabicSentimentClassifier(args.model_path)

    if args.text:
        clf.pretty_print(args.text)

    elif args.file:
        lines = Path(args.file).read_text(encoding='utf-8').strip().splitlines()
        lines = [l.strip() for l in lines if l.strip()]
        results = clf.predict_batch(lines)
        for r in results:
            print(f"{r['emoji']} {r['label']:<10} ({r['confidence']*100:.1f}%)  →  {r['text']}")

    else:
        # Demo mode — runs built-in examples
        demo_texts = [
            "المنتج ده كان ممتاز جداً وسريع في التوصيل",
            "خدمة العملاء بتاعتهم بطيئة جداً ومش مفيدة",
            "المنتج عادي مش أحسن ولا أسوأ",
            "سيء جداً لن أشتري مرة أخرى",
            "تجربة رائعة وسعر مناسب",
        ]
        print("\n" + "="*60)
        print("  Arabic Sentiment Classifier — Demo")
        print("="*60)
        for text in demo_texts:
            clf.pretty_print(text)
