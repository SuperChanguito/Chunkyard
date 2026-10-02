"""Run Chunkyard with the tests' fake embedder (no PyTorch, no model download).

Used by qa_ui_check.js. From the project folder:
    uv run python tests/ui/preview_server.py [port]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

import uvicorn  # noqa: E402

import chunkyard.app as app_module  # noqa: E402
from chunkyard.engine import Store  # noqa: E402
from test_chunkyard import FakeEmbedder  # noqa: E402


class PreviewEmbedder(FakeEmbedder):
    name = "preview (fake embedder)"
    error = None

    def load_async(self):
        pass


app_module.embedder = PreviewEmbedder()
app_module.store = Store(app_module.embedder)
port = int(sys.argv[1]) if len(sys.argv) > 1 else 8766
uvicorn.run(app_module.app, host="127.0.0.1", port=port, log_level="warning")
