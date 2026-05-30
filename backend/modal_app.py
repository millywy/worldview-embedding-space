"""Modal deployment entrypoint.

Deploy from the repository root:
    modal deploy backend/modal_app.py

If you have an OpenRouter key, create a Modal secret first and deploy with:
    modal secret create openrouter-api-key OPENROUTER_API_KEY=...
    OPENROUTER_MODAL_SECRET=openrouter-api-key modal deploy backend/modal_app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[1]
SECRET_NAME = os.environ.get("OPENROUTER_MODAL_SECRET", "")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(ROOT / "backend" / "requirements.txt"))
    .add_local_dir(ROOT / "backend", remote_path="/root/backend", copy=True)
    .add_local_dir(ROOT / "data", remote_path="/root/worldview_data", copy=True)
)

app = modal.App("worldview-embedding-space")
secrets = [modal.Secret.from_name(SECRET_NAME)] if SECRET_NAME else []


@app.function(image=image, secrets=secrets, timeout=240)
@modal.asgi_app()
def fastapi_app():
    import os
    import sys

    os.environ["WORLDVIEW_DATA_DIR"] = "/root/worldview_data"
    sys.path.insert(0, "/root")

    from backend.api import app as fastapi_app_instance

    return fastapi_app_instance
