"""Utility tools for the chat agent."""

import csv
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    """Return project root assuming this file is under `src/`."""
    return Path(__file__).resolve().parent.parent


def load_faqs() -> list[dict[str, str]]:
    """
    Load FAQs from `data/faq.csv`.

    The CSV is expected to have headers: `id,Question,Answer`.
    """
    logger.info("project root: %s", _project_root())
    data_path = _project_root() / "data" / "faq.csv"
    if not data_path.exists():
        return []
    with data_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [
            {
                "id": str(row.get("id", "")).strip(),
                "question": str(row.get("Question", "")).strip(),
                "answer": str(row.get("Answer", "")).strip(),
            }
            for row in reader
        ]


FAQS: list[dict[str, str]] = load_faqs()


def _embeddings_csv_path() -> Path:
    return _project_root() / "data" / "faq-embedding.csv"


def _to_cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors without numpy."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _build_embeddings_if_needed() -> list[dict[str, Any]]:
    """
    Ensure `data/faq-embedding.csv` exists.

    If missing, create it by computing embeddings for current FAQs using
    `text-embedding-3-small`.

    Returns loaded rows with parsed embeddings.
    """
    target = _embeddings_csv_path()
    if target.exists():
        return _load_embeddings_csv(target)

    if not FAQS:
        return []

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
    )
    texts = [f"Q: {f['question']}\nA: {f['answer']}" for f in FAQS]
    vectors = embeddings.embed_documents(texts)

    rows: list[dict[str, Any]] = []
    for faq, embedding in zip(FAQS, vectors, strict=True):
        rows.append(
            {
                "id": faq["id"],
                "question": faq["question"],
                "answer": faq["answer"],
                "embedding": embedding,
            }
        )

    # Save CSV
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["id", "question", "answer", "embedding"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "id": r["id"],
                    "question": r["question"],
                    "answer": r["answer"],
                    "embedding": json.dumps(r["embedding"], ensure_ascii=False),
                }
            )

    return rows


def _load_embeddings_csv(path: Path) -> list[dict[str, Any]]:
    """Load embedding rows from CSV and parse embeddings from JSON string."""
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, Any]] = []
        for row in reader:
            emb_raw = row.get("embedding", "[]")
            emb_parsed: Any
            if isinstance(emb_raw, str):
                try:
                    emb_parsed = json.loads(emb_raw)
                except (json.JSONDecodeError, TypeError, ValueError):
                    # Skip malformed rows
                    continue
            else:
                emb_parsed = emb_raw

            rows.append(
                {
                    "id": str(row.get("id", "")),
                    "question": str(row.get("question", "")),
                    "answer": str(row.get("answer", "")),
                    "embedding": emb_parsed,
                }
            )
        return rows


def _load_or_create_embeddings() -> list[dict[str, Any]]:
    path = _embeddings_csv_path()
    if path.exists():
        return _load_embeddings_csv(path)
    return _build_embeddings_if_needed()


@tool
def search_faqs(query: str) -> list[dict[str, str]]:
    """FAQの簡易検索。質問または回答に部分一致する行を返します。"""
    normalized = (query or "").lower()
    if not normalized:
        return []
    return [
        faq
        for faq in FAQS
        if normalized in faq["question"].lower() or normalized in faq["answer"].lower()
    ]


@tool
def semantic_search_faqs(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """
    OpenAI埋め込みを用いた簡易ベクトル検索。

    `data/faq-embedding.csv` が無ければ生成します。

    返却は上位 `top_k` 件の {id, question, answer, score}。
    """
    q = (query or "").strip()
    if not q:
        return []

    rows = _load_or_create_embeddings()
    if not rows:
        return []

    # Embed query
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
    )
    q_vec = embeddings.embed_query(q)

    # Score by cosine similarity
    scored = []
    for r in rows:
        emb = r.get("embedding")
        if not isinstance(emb, list):
            continue
        score = _to_cosine_similarity(q_vec, emb)
        scored.append(
            {
                "id": r["id"],
                "question": r["question"],
                "answer": r["answer"],
                "score": score,
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[: max(1, int(top_k))]
