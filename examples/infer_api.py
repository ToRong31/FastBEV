from __future__ import annotations

import argparse
import json
import sys
from contextlib import ExitStack
from pathlib import Path

import requests


CAMERA_DEFAULTS = {
    "front": "0-FRONT.jpg",
    "front_right": "1-FRONT_RIGHT.jpg",
    "front_left": "2-FRONT_LEFT.jpg",
    "back": "3-BACK.jpg",
    "back_left": "4-BACK_LEFT.jpg",
    "back_right": "5-BACK_RIGHT.jpg",
}

TENSOR_DEFAULTS = {
    "valid_c_idx": "valid_c_idx.tensor",
    "x": "x.tensor",
    "y": "y.tensor",
}


def _resolve_path(data_dir: Path, explicit: str | None, default_name: str) -> Path:
    return Path(explicit).expanduser() if explicit else data_dir / default_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call the FastBEV /infer API with six images and geometry tensors.")
    parser.add_argument("--url", default="http://localhost:8080/infer", help="FastAPI /infer endpoint URL.")
    parser.add_argument("--data-dir", default="example-data", help="Directory containing default sample inputs.")
    parser.add_argument("--output", help="Optional path to write the JSON response.")
    parser.add_argument("--timeout", type=float, default=300.0, help="Request timeout in seconds.")

    for field, default_name in CAMERA_DEFAULTS.items():
        parser.add_argument(f"--{field.replace('_', '-')}", help=f"Path for {field}; default: DATA_DIR/{default_name}")
    for field, default_name in TENSOR_DEFAULTS.items():
        parser.add_argument(f"--{field.replace('_', '-')}", help=f"Path for {field}; default: DATA_DIR/{default_name}")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser()

    input_paths = {}
    for field, default_name in CAMERA_DEFAULTS.items():
        input_paths[field] = _resolve_path(data_dir, getattr(args, field), default_name)
    for field, default_name in TENSOR_DEFAULTS.items():
        input_paths[field] = _resolve_path(data_dir, getattr(args, field), default_name)

    missing = [f"{field}: {path}" for field, path in input_paths.items() if not path.exists()]
    if missing:
        print("Missing input files:", file=sys.stderr)
        for item in missing:
            print(f"  {item}", file=sys.stderr)
        return 2

    with ExitStack() as stack:
        files = {}
        for field, path in input_paths.items():
            mime = "image/jpeg" if field in CAMERA_DEFAULTS else "application/octet-stream"
            files[field] = (path.name, stack.enter_context(path.open("rb")), mime)

        response = requests.post(args.url, files=files, timeout=args.timeout)
        if response.status_code >= 400:
            print(f"Request failed: HTTP {response.status_code}", file=sys.stderr)
            print(response.text, file=sys.stderr)
            return 1

    payload = response.json()
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
