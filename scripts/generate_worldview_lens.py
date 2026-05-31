"""Generate topic and residual-worldview artifacts from the current corpus.

Inspect topic clusters:
    modal run scripts/generate_worldview_lens.py

Generate artifacts after reviewing TOPIC_LABELS:
    modal run scripts/generate_worldview_lens.py --generate
"""

from __future__ import annotations

import io
import json
import pickle
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[1]
N_TOPICS = 8

# Review with the default inspect command before generating artifacts.
TOPIC_LABELS = {
    0: "Health and Public Welfare",
    1: "Culture, Religion and Media",
    2: "Sports and Politics",
    3: "Courts and Accountability",
    4: "Foreign Policy and Conflict",
    5: "Immigration and Enforcement",
    6: "Elections and Party Politics",
    7: "Trump and Governance",
}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy==2.2.1",
        "requests==2.32.3",
        "scikit-learn==1.6.0",
        "umap-learn==0.5.7",
    )
    .add_local_dir(ROOT / "data", remote_path="/root/worldview_data", copy=True)
)

app = modal.App("worldview-lens-generator")
openrouter_secret = modal.Secret.from_name("openrouter-api-key")


def _normalize(matrix):
    import numpy as np

    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.maximum(norms, 1e-9)


def _cluster_corpus():
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    data_dir = Path("/root/worldview_data")
    points = json.loads((data_dir / "points.json").read_text(encoding="utf-8"))
    articles = json.loads((data_dir / "articles.json").read_text(encoding="utf-8"))
    embeddings = np.load(data_dir / "embeddings.npy").astype("float32")
    norm_embeddings = _normalize(embeddings)

    topic_model = KMeans(n_clusters=N_TOPICS, random_state=42, n_init=30)
    topic_ids = topic_model.fit_predict(norm_embeddings)
    topic_centroids = _normalize(topic_model.cluster_centers_.astype("float32"))

    texts = [
        f"{article.get('headline', '')}. {article.get('text', '')[:1800]}"
        for article in articles
    ]
    tfidf = TfidfVectorizer(
        stop_words="english",
        max_df=0.84,
        min_df=2,
        ngram_range=(1, 2),
        max_features=4000,
    )
    matrix = tfidf.fit_transform(texts)
    vocabulary = tfidf.get_feature_names_out()

    topic_summaries = []
    for topic_id in range(N_TOPICS):
        indices = np.flatnonzero(topic_ids == topic_id)
        term_scores = np.asarray(matrix[indices].mean(axis=0)).ravel()
        top_term_indices = term_scores.argsort()[::-1][:12]
        top_terms = [str(vocabulary[index]) for index in top_term_indices if term_scores[index] > 0]

        centroid = topic_centroids[topic_id]
        similarities = norm_embeddings[indices] @ centroid
        representative = indices[np.argsort(-similarities)[:6]]
        topic_summaries.append(
            {
                "id": int(topic_id),
                "count": int(len(indices)),
                "top_terms": top_terms,
                "headlines": [articles[int(index)]["headline"] for index in representative],
            }
        )

    return {
        "points": points,
        "articles": articles,
        "norm_embeddings": norm_embeddings,
        "topic_ids": topic_ids,
        "topic_centroids": topic_centroids,
        "summaries": topic_summaries,
    }


@app.function(image=image, timeout=240)
def inspect_clusters():
    return _cluster_corpus()["summaries"]


@app.function(image=image, secrets=[openrouter_secret], timeout=240)
def suggest_labels():
    import os

    import requests

    summaries = _cluster_corpus()["summaries"]
    prompt = f"""
Name each political-news topic cluster using a restrained 2-5 word label.
Use broad subject matter, not ideological assumptions. Return only JSON mapping integer ids
to labels. Clusters:
{json.dumps(summaries, indent=2)}
""".strip()
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
            "X-Title": "Worldview Lens Generator",
        },
        json={
            "model": "openai/gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "You create concise, neutral information-architecture labels.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 240,
            "response_format": {"type": "json_object"},
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


@app.function(image=image, timeout=360)
def generate_artifacts():
    import numpy as np
    import umap

    clustered = _cluster_corpus()
    points = clustered["points"]
    articles = clustered["articles"]
    norm_embeddings = clustered["norm_embeddings"]
    topic_ids = clustered["topic_ids"]
    topic_centroids = clustered["topic_centroids"]
    summaries = clustered["summaries"]

    residual_embeddings = _normalize(norm_embeddings - topic_centroids[topic_ids]).astype("float32")
    worldview_model = umap.UMAP(
        n_components=3,
        n_neighbors=18,
        min_dist=0.18,
        metric="cosine",
        random_state=42,
    )
    worldview_coords = worldview_model.fit_transform(residual_embeddings).astype("float32")

    enriched_points = []
    worldview_points = []
    for index, point in enumerate(points):
        topic_id = int(topic_ids[index])
        topic_label = TOPIC_LABELS[topic_id]
        enriched = {
            **point,
            "topic_id": topic_id,
            "topic_label": topic_label,
        }
        enriched_points.append(enriched)
        worldview_points.append(
            {
                **enriched,
                "x": float(worldview_coords[index, 0]),
                "y": float(worldview_coords[index, 1]),
                "z": float(worldview_coords[index, 2]),
            }
        )

    outlets = sorted({point["outlet"] for point in points})
    worldview_centroids = {}
    for outlet in outlets:
        indices = [index for index, point in enumerate(points) if point["outlet"] == outlet]
        coords = worldview_coords[indices]
        worldview_centroids[outlet] = {
            "x": float(coords[:, 0].mean()),
            "y": float(coords[:, 1].mean()),
            "z": float(coords[:, 2].mean()),
            "count": len(indices),
        }

    editorial_points = np.array(
        [[point["x"], point["y"], point["z"]] for point in points],
        dtype="float32",
    )
    topics = []
    for summary in summaries:
        topic_id = summary["id"]
        indices = np.flatnonzero(topic_ids == topic_id)
        editorial_center = editorial_points[indices].mean(axis=0)
        worldview_center = worldview_coords[indices].mean(axis=0)
        topics.append(
            {
                **summary,
                "label": TOPIC_LABELS[topic_id],
                "editorial": {
                    "x": float(editorial_center[0]),
                    "y": float(editorial_center[1]),
                    "z": float(editorial_center[2]),
                },
                "worldview": {
                    "x": float(worldview_center[0]),
                    "y": float(worldview_center[1]),
                    "z": float(worldview_center[2]),
                },
            }
        )

    def json_bytes(value):
        return json.dumps(value, indent=2, ensure_ascii=False).encode("utf-8")

    def npy_bytes(value):
        buffer = io.BytesIO()
        np.save(buffer, value)
        return buffer.getvalue()

    return {
        "points.json": json_bytes(enriched_points),
        "worldview_points.json": json_bytes(worldview_points),
        "worldview_centroids.json": json_bytes(worldview_centroids),
        "topics.json": json_bytes(topics),
        "topic_centroids.npy": npy_bytes(topic_centroids.astype("float32")),
        "topic_ids.npy": npy_bytes(topic_ids.astype("int32")),
        "residual_embeddings.npy": npy_bytes(residual_embeddings),
        "worldview_umap_model.pkl": pickle.dumps(worldview_model),
        "articles.json": json_bytes(
            [
                {
                    **article,
                    "topic_id": int(topic_ids[index]),
                    "topic_label": TOPIC_LABELS[int(topic_ids[index])],
                }
                for index, article in enumerate(articles)
            ]
        ),
    }


@app.local_entrypoint()
def main(generate: bool = False, suggest: bool = False):
    if suggest:
        print(suggest_labels.remote())
        return

    if not generate:
        print(json.dumps(inspect_clusters.remote(), indent=2, ensure_ascii=False))
        return

    artifacts = generate_artifacts.remote()
    data_dir = ROOT / "data"
    public_dir = ROOT / "public" / "worldview"
    data_dir.mkdir(exist_ok=True)
    public_dir.mkdir(parents=True, exist_ok=True)

    data_files = {
        "points.json",
        "articles.json",
        "worldview_points.json",
        "worldview_centroids.json",
        "topics.json",
        "topic_centroids.npy",
        "topic_ids.npy",
        "residual_embeddings.npy",
        "worldview_umap_model.pkl",
    }
    public_files = {
        "points.json",
        "worldview_points.json",
        "worldview_centroids.json",
        "topics.json",
    }

    for name in data_files:
        (data_dir / name).write_bytes(artifacts[name])
    for name in public_files:
        (public_dir / name).write_bytes(artifacts[name])

    print("Generated:")
    for name in sorted(data_files | public_files):
        print(f"- {name}")
