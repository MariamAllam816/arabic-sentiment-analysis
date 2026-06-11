"""
preprocess.py
Arabic text cleaning and normalization for sentiment analysis.
Handles MSA (Modern Standard Arabic) and Egyptian/Gulf dialect tweets.
"""

import re
import unicodedata
from typing import Optional


# ─── Arabic Unicode ranges ────────────────────────────────────────────────────
ARABIC_LETTERS = r'\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF'
DIACRITICS = r'[\u064B-\u065F\u0670]'                 # Fatha, Damma, Kasra, etc.
TATWEEL = r'\u0640'                                   # ـ  elongation character


# ─── Normalization maps ───────────────────────────────────────────────────────
ALEF_VARIANTS = {
    '\u0622': '\u0627',   # آ → ا
    '\u0623': '\u0627',   # أ → ا
    '\u0625': '\u0627',   # إ → ا
    '\u0671': '\u0627',   # ٱ → ا
}

YA_VARIANTS = {
    '\u0649': '\u064A',   # ى → ي  (Alef Maqsura)
}

HA_VARIANTS = {
    '\u0629': '\u0647',   # ة → ه  (Ta Marbuta → Ha)  — optional, see arg
}


# ─── Core cleaner ─────────────────────────────────────────────────────────────

class ArabicPreprocessor:
    """
    Cleans and normalizes Arabic text for NLP tasks.

    Args:
        normalize_alef (bool): Normalize Alef variants to bare Alef (ا). Default True.
        normalize_ya (bool): Normalize Alef Maqsura (ى) to Ya (ي). Default True.
        normalize_ha (bool): Normalize Ta Marbuta (ة) to Ha (ه). Default False.
        remove_diacritics (bool): Strip tashkeel. Default True.
        remove_tatweel (bool): Remove elongation character (ـ). Default True.
        remove_punctuation (bool): Remove Arabic & Latin punctuation. Default True.
        remove_latin (bool): Remove Latin characters (keeps Arabic + numbers). Default False.
        remove_numbers (bool): Remove digits. Default False.
        remove_urls (bool): Strip http/https URLs. Default True.
        remove_mentions (bool): Strip @mentions. Default True.
        remove_hashtags (bool): Strip #hashtags or keep the word part. Default False.
        strip_hashtag_symbol (bool): Keep word after #, remove just the #. Default True.
        min_length (int): Drop tokens shorter than this after cleaning. Default 2.
    """

    def __init__(
        self,
        normalize_alef: bool = True,
        normalize_ya: bool = True,
        normalize_ha: bool = False,
        remove_diacritics: bool = True,
        remove_tatweel: bool = True,
        remove_punctuation: bool = True,
        remove_latin: bool = False,
        remove_numbers: bool = False,
        remove_urls: bool = True,
        remove_mentions: bool = True,
        remove_hashtags: bool = False,
        strip_hashtag_symbol: bool = True,
        min_length: int = 2,
    ):
        self.normalize_alef = normalize_alef
        self.normalize_ya = normalize_ya
        self.normalize_ha = normalize_ha
        self.remove_diacritics = remove_diacritics
        self.remove_tatweel = remove_tatweel
        self.remove_punctuation = remove_punctuation
        self.remove_latin = remove_latin
        self.remove_numbers = remove_numbers
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.strip_hashtag_symbol = strip_hashtag_symbol
        self.min_length = min_length

    # ── private helpers ───────────────────────────────────────────────────────

    def _remove_urls(self, text: str) -> str:
        return re.sub(r'https?://\S+|www\.\S+', ' ', text)

    def _remove_mentions(self, text: str) -> str:
        return re.sub(r'@\w+', ' ', text)

    def _handle_hashtags(self, text: str) -> str:
        if self.remove_hashtags:
            return re.sub(r'#\w+', ' ', text)
        if self.strip_hashtag_symbol:
            return re.sub(r'#(\w+)', r'\1', text)
        return text

    def _remove_diacritics(self, text: str) -> str:
        return re.sub(DIACRITICS, '', text)

    def _remove_tatweel(self, text: str) -> str:
        return text.replace(TATWEEL, '')

    def _normalize_chars(self, text: str) -> str:
        if self.normalize_alef:
            for variant, base in ALEF_VARIANTS.items():
                text = text.replace(variant, base)
        if self.normalize_ya:
            for variant, base in YA_VARIANTS.items():
                text = text.replace(variant, base)
        if self.normalize_ha:
            for variant, base in HA_VARIANTS.items():
                text = text.replace(variant, base)
        return text

    def _remove_punctuation(self, text: str) -> str:
        # Arabic punctuation + standard ASCII punctuation
        text = re.sub(r'[،؛؟!٪«»\(\)\[\]\{\}\"\'`.,;:!?/\\|@#$%^&*_+=<>~-]', ' ', text)
        return text

    def _remove_latin(self, text: str) -> str:
        return re.sub(r'[a-zA-Z]', ' ', text)

    def _remove_numbers(self, text: str) -> str:
        # Remove both Arabic-Indic (٠١٢...) and Western digits
        return re.sub(r'[\d٠-٩]+', ' ', text)

    def _remove_emojis(self, text: str) -> str:
        # Remove all non-Arabic, non-Latin, non-digit characters (catches emojis)
        return re.sub(
            r'[^\u0600-\u06FF\u0750-\u077F\u0041-\u007A\u0030-\u0039\s]', ' ', text
        )

    def _collapse_whitespace(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text).strip()

    # ── public API ────────────────────────────────────────────────────────────

    def clean(self, text: str) -> str:
        """
        Apply the full preprocessing pipeline to a single string.

        Args:
            text: Raw Arabic text (tweet, review, etc.)

        Returns:
            Cleaned, normalized string.
        """
        if not isinstance(text, str) or not text.strip():
            return ''

        if self.remove_urls:
            text = self._remove_urls(text)
        if self.remove_mentions:
            text = self._remove_mentions(text)

        text = self._handle_hashtags(text)

        if self.remove_diacritics:
            text = self._remove_diacritics(text)
        if self.remove_tatweel:
            text = self._remove_tatweel(text)

        text = self._normalize_chars(text)
        text = self._remove_emojis(text)

        if self.remove_punctuation:
            text = self._remove_punctuation(text)
        if self.remove_latin:
            text = self._remove_latin(text)
        if self.remove_numbers:
            text = self._remove_numbers(text)

        text = self._collapse_whitespace(text)
        return text

    def clean_batch(self, texts: list[str]) -> list[str]:
        """Clean a list of texts."""
        return [self.clean(t) for t in texts]


# ─── Label helpers ────────────────────────────────────────────────────────────

LABEL2ID = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def encode_label(label: str) -> int:
    """Map string label → integer id."""
    return LABEL2ID[label]


def decode_label(label_id: int) -> str:
    """Map integer id → string label."""
    return ID2LABEL[label_id]


# ─── Quick smoke test ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    preprocessor = ArabicPreprocessor()

    samples = [
        "المنتج ده كان ممتاز جداً وسريع في التوصيل 🚀",
        "خدمة العملاء بطيئة جداً @support_team https://example.com",
        "#تجربة_سيئة الطلب وصل متأخر وغلط",
        "كويس ومعقول",
        "سيء جداً لن أشتري مرة أخرى!!!",
    ]

    print("=" * 60)
    print("Arabic Preprocessor — Smoke Test")
    print("=" * 60)
    for text in samples:
        cleaned = preprocessor.clean(text)
        print(f"  RAW    : {text}")
        print(f"  CLEANED: {cleaned}")
        print()
