"""FastAPI service for projecting user writing into worldview space."""

from __future__ import annotations

import json
import os
import pickle
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("WORLDVIEW_DATA_DIR", ROOT / "data"))
MODEL_NAME = os.environ.get("WORLDVIEW_MODEL", "all-MiniLM-L6-v2")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")

STOP_WORDS = {
    "about",
    "after",
    "again",
    "against",
    "also",
    "because",
    "before",
    "being",
    "between",
    "could",
    "during",
    "every",
    "from",
    "have",
    "into",
    "more",
    "most",
    "over",
    "said",
    "should",
    "than",
    "that",
    "their",
    "there",
    "these",
    "they",
    "this",
    "through",
    "under",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
}

app = FastAPI(title="Worldview Embedding Space API", version="0.1.0")

cors_origins = [
    origin.strip()
    for origin in os.environ.get("CORS_ORIGINS", "*").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextInput(BaseModel):
    text: str = Field(min_length=20, max_length=12000)
    interpret: bool = True


def _load_json(name: str) -> Any:
    with (DATA_DIR / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.maximum(norms, 1e-9)


@lru_cache(maxsize=1)
def get_data() -> dict[str, Any]:
    points = _load_json("points.json")
    articles = _load_json("articles.json")
    centroids_3d = _load_json("centroids.json")
    embeddings = np.load(DATA_DIR / "embeddings.npy").astype("float32")
    norm_embeddings = _normalize(embeddings)

    with (DATA_DIR / "umap_model.pkl").open("rb") as handle:
        umap_model = pickle.load(handle)

    outlet_indices: dict[str, list[int]] = defaultdict(list)
    for index, point in enumerate(points):
        outlet_indices[point["outlet"]].append(index)

    outlet_centroids = {}
    for outlet, indices in outlet_indices.items():
        outlet_centroids[outlet] = _normalize(
            norm_embeddings[indices].mean(axis=0, keepdims=True)
        )[0]

    return {
        "points": points,
        "articles": articles,
        "centroids_3d": centroids_3d,
        "embeddings": embeddings,
        "norm_embeddings": norm_embeddings,
        "outlet_centroids": outlet_centroids,
        "umap_model": umap_model,
    }


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def extract_terms(text: str, limit: int = 8) -> list[str]:
    words = re.findall(r"[a-z][a-z-]{3,}", text.lower())
    counts = Counter(word for word in words if word not in STOP_WORDS)
    return [word for word, _ in counts.most_common(limit)]


def outlet_similarities(embedding: np.ndarray, outlet_centroids: dict[str, np.ndarray]) -> dict[str, float]:
    normalized = _normalize(embedding.reshape(1, -1))[0]
    raw_scores = {
        outlet: float(np.dot(normalized, centroid))
        for outlet, centroid in outlet_centroids.items()
    }
    max_score = max(raw_scores.values())
    exp_scores = {
        outlet: float(np.exp((score - max_score) * 12.0))
        for outlet, score in raw_scores.items()
    }
    total = sum(exp_scores.values()) or 1.0
    return {
        outlet: round(score / total * 100, 1)
        for outlet, score in sorted(exp_scores.items(), key=lambda item: item[1], reverse=True)
    }


def nearest_articles(embedding: np.ndarray, data: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    normalized = _normalize(embedding.reshape(1, -1))[0]
    similarities = data["norm_embeddings"] @ normalized
    nearest_indices = np.argsort(-similarities)[:limit]
    nearest = []
    for index in nearest_indices:
        point = data["points"][int(index)]
        nearest.append(
            {
                "headline": point["headline"],
                "outlet": point["outlet"],
                "url": point["url"],
                "text_preview": point.get("text_preview", ""),
                "similarity": round(float(similarities[index]) * 100, 1),
            }
        )
    return nearest


def project_embedding(embedding: np.ndarray, data: dict[str, Any]) -> tuple[np.ndarray, str]:
    """Project to 3D, with a stable fallback if UMAP's numba path fails."""
    try:
        coords = data["umap_model"].transform(embedding.reshape(1, -1))[0]
        return np.asarray(coords, dtype="float32"), "umap"
    except Exception:
        normalized = _normalize(embedding.reshape(1, -1))[0]
        similarities = data["norm_embeddings"] @ normalized
        nearest_indices = np.argsort(-similarities)[:12]
        top_scores = similarities[nearest_indices]
        weights = np.exp((top_scores - top_scores.max()) * 18.0)
        weights = weights / max(float(weights.sum()), 1e-9)

        coords = np.array(
            [
                [
                    data["points"][int(index)]["x"],
                    data["points"][int(index)]["y"],
                    data["points"][int(index)]["z"],
                ]
                for index in nearest_indices
            ],
            dtype="float32",
        )
        return (coords * weights[:, None]).sum(axis=0), "nearest-neighbor-projection"


def interpret_with_openrouter(text: str, result: dict[str, Any]) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return ""

    nearest_lines = "\n".join(
        f"- {item['outlet']}: {item['headline']}" for item in result["nearest"][:4]
    )
    top_outlet = result["top_outlet"]
    terms = ", ".join(result.get("distinctive_terms", [])[:8]) or "none"

    prompt = f"""
You are analyzing a political-writing embedding demo for a class on liberty and algorithms.
The user's writing was placed nearest to {top_outlet}.
Similarity distribution: {json.dumps(result["similarities"])}
Distinctive terms: {terms}
Nearest articles:
{nearest_lines}

Write 3 concise sentences explaining what framing, word choice, or institutional assumptions
may have pulled the text toward that outlet. Avoid claiming the model knows the user's true
beliefs. End with one sentence connecting this visibility to algorithmic liberty.
""".strip()

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.environ.get("OPENROUTER_SITE_URL", "http://localhost:5173"),
            "X-Title": "Worldview Embedding Space",
        },
        json={
            "model": OPENROUTER_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are precise, nonpartisan, and careful about model limitations.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.45,
            "max_tokens": 180,
        },
        timeout=22,
    )
    response.raise_for_status()
    payload = response.json()
    return payload["choices"][0]["message"]["content"].strip()


@app.get("/health")
def health() -> dict[str, Any]:
    data = get_data()
    return {
        "ok": True,
        "points": len(data["points"]),
        "outlets": sorted(data["outlet_centroids"].keys()),
        "model": MODEL_NAME,
        "openrouter": bool(os.environ.get("OPENROUTER_API_KEY")),
    }


@app.get("/points")
def get_points() -> list[dict[str, Any]]:
    return get_data()["points"]


@app.get("/centroids")
def get_centroids() -> dict[str, Any]:
    return get_data()["centroids_3d"]


@app.post("/embed")
def embed_text(input_data: TextInput) -> dict[str, Any]:
    text = input_data.text.strip()
    data = get_data()
    model = get_model()

    embedding = model.encode([text[:6000]], convert_to_numpy=True)[0].astype("float32")
    coords, projection = project_embedding(embedding, data)
    similarities = outlet_similarities(embedding, data["outlet_centroids"])
    top_outlet, top_score = next(iter(similarities.items()))

    result = {
        "x": float(coords[0]),
        "y": float(coords[1]),
        "z": float(coords[2]),
        "similarities": similarities,
        "top_outlet": top_outlet,
        "top_score": top_score,
        "nearest": nearest_articles(embedding, data),
        "distinctive_terms": extract_terms(text),
        "mode": "modal-backend",
        "projection": projection,
    }

    if input_data.interpret:
        try:
            result["interpretation"] = interpret_with_openrouter(text, result)
        except Exception as error:  # noqa: BLE001 - surface provider failures without failing embed.
            result["interpretation"] = ""
            result["interpretation_error"] = str(error)

    return result
