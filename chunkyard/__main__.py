"""Run with:  uv run python -m chunkyard  [--port 8000] [--no-browser]"""

import argparse
import os
import threading
import webbrowser

import uvicorn


def main() -> None:
    # Windows without Developer Mode can't make symlinks; the model cache still works.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    parser = argparse.ArgumentParser(prog="chunkyard")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    url = f"http://127.0.0.1:{args.port}"
    print(f"Chunkyard running at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, [url]).start()
    uvicorn.run("chunkyard.app:app", host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
