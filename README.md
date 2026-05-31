# Worldview Embedding Space

Interactive 3D visualization that places political writing into a worldview embedding space.

The current dataset is an evolving seed corpus:

- 284 article points across Fox News, Breitbart, NYT, The Guardian, NBC News, Washington Post, and NPR
- `all-MiniLM-L6-v2` 384-dimensional embeddings
- fitted UMAP model for projecting new text into the same 3D space
- outlet centroids and raw article metadata

The interface has two lenses:

- `Worldview`: residual framing coordinates after subtracting the detected topic centroid
- `Landscape`: the original semantic embedding projection, where topic and editorial framing coexist

The app is intentionally structured so the corpus can be regenerated later with more articles.

## Local Frontend

```bash
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

Without `VITE_API_BASE_URL`, the frontend uses a bundled demo-mode placement so the public Vercel site still works.

## Local Backend

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.api:app --reload --port 8000
```

Then set:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
```

## OpenRouter Interpretation

Set `OPENROUTER_API_KEY` on the backend. Optional:

```bash
OPENROUTER_MODEL=openai/gpt-4o-mini
```

If the key is missing, `/embed` still returns coordinates, similarities, and nearest articles; it just omits the LLM interpretation.

## Modal Backend Deployment

From the repo root:

```bash
modal deploy backend/modal_app.py
```

With OpenRouter:

```bash
modal secret create openrouter-api-key OPENROUTER_API_KEY=your_key_here
modal deploy backend/modal_app.py
```

Use the Modal endpoint URL as `VITE_API_BASE_URL` for Vercel.

## Vercel Frontend Deployment

```bash
npx vercel --yes
```

In Vercel, set `VITE_API_BASE_URL` to the Modal FastAPI URL if you want real sentence-transformer embedding. Leave it blank for the bundled public demo mode.

## Regenerating The Corpus Later

After replacing the base corpus artifacts, inspect topic clusters:

```bash
modal run scripts/generate_worldview_lens.py
```

Optionally ask OpenRouter for neutral topic-label suggestions:

```bash
modal run scripts/generate_worldview_lens.py --suggest
```

Review `TOPIC_LABELS` in `scripts/generate_worldview_lens.py`, then generate the derived worldview artifacts:

```bash
modal run scripts/generate_worldview_lens.py --generate
```

The complete artifact contract is:

- `data/points.json`
- `data/centroids.json`
- `data/articles.json`
- `data/embeddings.npy`
- `data/umap_model.pkl`
- `data/topics.json`
- `data/topic_centroids.npy`
- `data/topic_ids.npy`
- `data/residual_embeddings.npy`
- `data/worldview_points.json`
- `data/worldview_centroids.json`
- `data/worldview_umap_model.pkl`
- `public/worldview/points.json`
- `public/worldview/centroids.json`
- `public/worldview/topics.json`
- `public/worldview/worldview_points.json`
- `public/worldview/worldview_centroids.json`

The public frontend reads the JSON artifacts. The Modal backend uses the complete `data/` set.
