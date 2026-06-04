"""
retrieval_utils.py
------------------
InitERC-style TF-IDF demonstration retrieval from MELD training data.
Builds a local in-memory index at startup; zero API calls during inference.
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_CSV = os.path.join(BASE_DIR, "data", "train_sent_emo.csv")

# Emotion label set (MELD canonical)
MELD_EMOTIONS = ["neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"]

class MeldRetriever:
    """
    Lightweight TF-IDF index over MELD training utterances.
    Call get_top_k_examples(utterance, k=3) to get few-shot demonstrations.
    """

    def __init__(self, csv_path: str = TRAIN_CSV):
        print(f"[Retriever] Building TF-IDF index from: {csv_path}", flush=True)
        df = pd.read_csv(csv_path)

        # Normalise column names — handle both 'Emotion' and 'emotion'
        df.columns = [c.strip() for c in df.columns]
        emo_col = next((c for c in df.columns if c.lower() == "emotion"), None)
        utt_col = next((c for c in df.columns if c.lower() == "utterance"), None)

        if emo_col is None or utt_col is None:
            raise ValueError(
                f"[Retriever] Could not find Utterance/Emotion columns. "
                f"Available: {list(df.columns)}"
            )

        df = df[[utt_col, emo_col]].dropna()
        df[emo_col] = df[emo_col].str.lower().str.strip()

        # Only keep canonical MELD labels
        df = df[df[emo_col].isin(MELD_EMOTIONS)].reset_index(drop=True)

        self.utterances: list[str] = df[utt_col].tolist()
        self.labels: list[str] = df[emo_col].tolist()

        # Fit vectoriser
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=30_000,
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(self.utterances)
        print(
            f"[Retriever] Index ready: {len(self.utterances):,} examples, "
            f"vocab={self.vectorizer.vocabulary_.__len__():,}",
            flush=True,
        )

    def get_top_k_examples(
        self,
        query: str,
        k: int = 3,
        exclude_identical: bool = True,
    ) -> list[dict]:
        """
        Returns top-k most similar training examples to `query`.

        Returns
        -------
        list of dict with keys: 'utterance', 'emotion', 'score'
        Ordered by descending similarity.
        """
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix).flatten()

        # Argsort descending; skip index 0 if it's a perfect match
        sorted_indices = np.argsort(scores)[::-1]

        results = []
        for idx in sorted_indices:
            utt = self.utterances[idx]
            if exclude_identical and utt.strip().lower() == query.strip().lower():
                continue  # skip exact duplicates (test leakage)
            results.append({
                "utterance": utt,
                "emotion": self.labels[idx],
                "score": float(scores[idx]),
            })
            if len(results) >= k:
                break

        return results

    def format_for_prompt(self, examples: list[dict]) -> str:
        """
        Format retrieved examples as a numbered few-shot block for injection
        into the InitERC classifier prompt.
        """
        if not examples:
            return "[No similar examples found]"
        lines = []
        for i, ex in enumerate(examples, 1):
            lines.append(
                f'{i}. Utterance: "{ex["utterance"]}"  ->  Emotion: {ex["emotion"].upper()}'
            )
        return "\n".join(lines)


# Module-level singleton — instantiated once at import time
_retriever: MeldRetriever | None = None


def get_retriever() -> MeldRetriever:
    """Returns the module-level singleton, building it on first call."""
    global _retriever
    if _retriever is None:
        _retriever = MeldRetriever()
    return _retriever
