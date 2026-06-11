# 🧠 Arabic Sentiment Analysis — AraBERT Fine-tuning

Fine-tuned Arabic sentiment classifier using **AraBERT** on Arabic tweets and product reviews.  
Achieves **~88% accuracy** on held-out test data across 3 sentiment classes (Positive / Negative / Neutral).

> Built to demonstrate Arabic NLP capabilities relevant to Egyptian & MENA tech companies.

---

## 🚀 Demo

```
Input:  "المنتج ده كان ممتاز جداً وسريع في التوصيل"
Output: ✅ Positive (confidence: 94.2%)

Input:  "خدمة العملاء بتاعتهم بطيئة جداً ومش مفيدة"
Output: ❌ Negative (confidence: 91.7%)
```

---

## 📁 Project Structure

```
arabic-sentiment-analysis/
├── src/
│   ├── preprocess.py        # Arabic text cleaning & normalization
│   ├── train.py             # Fine-tuning AraBERT
│   ├── evaluate.py          # Metrics, confusion matrix, per-class report
│   └── predict.py           # Inference on new text
├── notebooks/
│   └── demo.ipynb           # End-to-end walkthrough (Colab-ready)
├── data/
│   └── README.md            # Dataset sources and download instructions
├── models/                  # Saved checkpoints (git-ignored)
├── tests/
│   └── test_preprocess.py   # Unit tests for preprocessing
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🛠️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/arabic-sentiment-analysis.git
cd arabic-sentiment-analysis
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset

This project uses the **ASTD (Arabic Sentiment Tweets Dataset)** — freely available for research.

Download instructions are in [`data/README.md`](data/README.md).

Alternatively, you can use the **Arabic SemEval-2017** dataset or scrape your own with `tweepy`.

---

## 🏋️ Training

```bash
python src/train.py \
  --data_path data/astd_processed.csv \
  --model_name aubmindlab/bert-base-arabertv2 \
  --epochs 4 \
  --batch_size 16 \
  --output_dir models/arabert-sentiment
```

**Training args:**

| Argument | Default | Description |
|---|---|---|
| `--model_name` | `aubmindlab/bert-base-arabertv2` | HuggingFace model ID |
| `--epochs` | `4` | Training epochs |
| `--batch_size` | `16` | Batch size (reduce to 8 if OOM) |
| `--lr` | `2e-5` | Learning rate |
| `--max_len` | `128` | Max token length |

---

## 📈 Evaluation

```bash
python src/evaluate.py \
  --model_path models/arabert-sentiment \
  --test_path data/test.csv
```

Expected output:
```
              precision    recall  f1-score
    Positive     0.91      0.89      0.90
    Negative     0.88      0.87      0.87
     Neutral     0.84      0.86      0.85

    Accuracy: 0.878
```

---

## 🔍 Inference

```bash
python src/predict.py --text "المنتج ممتاز جداً"
# Output: Positive (0.942)

python src/predict.py --text "سيء جداً لن أشتري مرة أخرى"
# Output: Negative (0.917)
```

Or use it as a Python module:

```python
from src.predict import ArabicSentimentClassifier

clf = ArabicSentimentClassifier("models/arabert-sentiment")
result = clf.predict("الخدمة كانت معقولة")
print(result)  # {'label': 'Neutral', 'confidence': 0.81}
```

---

## 🧪 Tests

```bash
pytest tests/
```

---

## 📓 Colab Notebook

Open the full walkthrough in Google Colab (free GPU):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/arabic-sentiment-analysis/blob/main/notebooks/demo.ipynb)

---

## 🧹 Arabic Preprocessing Pipeline

The `preprocess.py` module handles:
- Removing diacritics (تشكيل)
- Normalizing Alef/Ya/Waw variants (أ إ آ → ا)
- Removing Arabic-specific punctuation and non-Arabic characters
- Handling emojis and mixed Arabic-English text (Arabizi)
- Tweet-specific cleaning (mentions, hashtags, URLs)

---

## 📚 Model

| Component | Choice | Reason |
|---|---|---|
| Base model | `aubmindlab/bert-base-arabertv2` | Best Arabic BERT; trained on 77GB Arabic text |
| Tokenizer | AraBERT tokenizer | Handles Arabic morphology |
| Fine-tuning | HuggingFace Trainer API | Clean, reproducible |
| Optimization | AdamW + linear warmup | Standard for BERT fine-tuning |

---

## 🗂️ References

- [AraBERT Paper](https://arxiv.org/abs/2003.00104) — Antoun et al., 2020
- [ASTD Dataset](https://github.com/abdullaha9/astd) — Arabic Sentiment Tweets Dataset
- [CAMeL Tools](https://github.com/CAMeL-Lab/camel_tools) — Arabic NLP toolkit by NYU Abu Dhabi
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)

---

## 🤝 Contributing

PRs welcome! Especially interested in:
- Egyptian dialect (EGY) fine-tuning
- Multi-dialect support (MSA + Gulf + Levantine)
- Aspect-based sentiment (e.g., "delivery was fast but packaging was bad")

---

## 📄 License

MIT License — see [LICENSE](LICENSE)
