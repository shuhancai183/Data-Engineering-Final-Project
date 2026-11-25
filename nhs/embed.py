"""Compute and cache sentence embeddings for a task's rows and its query.

Embeddings are L2-normalised, so a dot product is a cosine similarity.

Usage:
  python -m nhs.embed --task 20ng_comp toxic_conv [--model NAME] [--batch-size 64] [--force]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from .paths import DATA, DEFAULT_MODEL, model_short


def load_texts(task: str) -> list[str]:
    with open(DATA / task / "texts.jsonl", encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f]


def load_meta(task: str) -> dict:
    with open(DATA / task / "meta.json", encoding="utf-8") as f:
        return json.load(f)


def get_encoder(model_name: str = DEFAULT_MODEL, device: str | None = None):
    import torch
    from sentence_transformers import SentenceTransformer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = 256
    return model


def encode(model, texts: list[str], batch_size: int = 64, progress: bool = True) -> np.ndarray:
    emb = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=progress,
    )
    return np.asarray(emb, dtype=np.float32)


def embed_task(task: str, model_name: str = DEFAULT_MODEL, batch_size: int = 64,
               force: bool = False, model=None):
    meta = load_meta(task)
    src = meta.get("emb_from") or task
    short = model_short(model_name)
    out = DATA / src / f"emb_{short}.npy"
    model = model or get_encoder(model_name)
    if force or not out.exists():
        texts = load_texts(src)
        t0 = time.time()
        emb = encode(model, texts, batch_size)
        np.save(out, emb)
        print(f"[{src}] embedded {len(texts)} rows -> {emb.shape} in {time.time() - t0:.0f}s", flush=True)
    else:
        print(f"[{src}] embeddings already cached ({out.name})", flush=True)
    q = encode(model, [meta["query"]], 1, progress=False)[0]
    np.save(DATA / task / f"query_emb_{short}.npy", q)
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", nargs="+", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    model = None
    for t in args.task:
        model = embed_task(t, args.model, args.batch_size, args.force, model)


if __name__ == "__main__":
    main()
