# Worldview Embedding Space

Interactive 3D visualization that places political writing into a worldview embedding space.

The current dataset is a v1 seed corpus:

- 263 article points across Fox News, Breitbart, NYT, The Guardian, NBC News, Washington Post, and NPR
- `all-MiniLM-L6-v2` 384-dimensional embeddings
- fitted UMAP model for projecting new text into the same 3D space
- outlet centroids and raw article metadata

The app is intentionally structured so the corpus can be regenerated later with more articles, then dropped back into `data/` and `public/worldview/`.

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

To upgrade the dataset, reproduce the same artifact contract:

- `data/points.json`
- `data/centroids.json`
- `data/articles.json`
- `data/embeddings.npy`
- `data/umap_model.pkl`
- `public/worldview/points.json`
- `public/worldview/centroids.json`

The frontend only needs `points.json` and `centroids.json`. The backend needs all five data artifacts.
