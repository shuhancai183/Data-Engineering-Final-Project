"""Convert raw datasets into the unified task format used by every experiment.

Each task directory ``data/<task>/`` contains:
  texts.jsonl     one {"row_id", "text"} object per line (absent when meta.emb_from points elsewhere)
  y.npy           int8 gold label: 1 positive, 0 negative, -1 unknown
  subtype.npy     int16 subtype id of each positive (-1 for negatives / unknown)
  meta.json       task description, query text and statistics
Optional:
  eval_mask.npy   bool, rows with expert labels that are reserved for evaluation only

Usage:
  python -m nhs.data.prepare --task 20ng_comp toxic_conv
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from collections import Counter

import numpy as np
import pandas as pd

from ..paths import DATA, RAW

QUERIES = {
    "20ng_comp": (
        "The post discusses computers: computer graphics, operating systems, "
        "PC or Mac hardware, or the X Window System."
    ),
    "toxic_conv": "The comment is toxic: rude, disrespectful, insulting, or hateful.",
    "jigsaw_threat": "The comment contains a threat of violence or harm against someone.",
    "jigsaw_severe": (
        "The comment contains a threat, identity-based hate, or severely toxic language."
    ),
    "lts_shark": (
        "The ad sells a product made from a real shark, ray or chimaera (e.g., jaws, sets of teeth, "
        "fins, skin). Exclude fossils, single teeth, images, replicas and synthetic items."
    ),
}


def save_task(name, texts, y, subtype, meta, eval_mask=None):
    d = DATA / name
    d.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y, dtype=np.int8)
    subtype = np.asarray(subtype, dtype=np.int16)
    assert len(y) == len(subtype)
    if meta.get("emb_from") in (None, name):
        assert len(texts) == len(y)
        with open(d / "texts.jsonl", "w", encoding="utf-8") as f:
            for i, t in enumerate(texts):
                f.write(json.dumps({"row_id": i, "text": t}, ensure_ascii=False) + "\n")
    np.save(d / "y.npy", y)
    np.save(d / "subtype.npy", subtype)
    if eval_mask is not None:
        np.save(d / "eval_mask.npy", np.asarray(eval_mask, dtype=bool))
    known = y >= 0
    names = meta.get("subtype_names") or []
    counts = Counter(subtype[y == 1].tolist())
    full = {
        "task_name": name,
        "query": QUERIES[name],
        **meta,
        "n_rows": int(len(y)),
        "n_gold_labeled": int(known.sum()),
        "n_pos": int((y == 1).sum()),
        "base_rate": float((y[known] == 1).mean()) if known.any() else None,
        "subtype_counts": {
            (names[k] if 0 <= k < len(names) else str(k)): int(v) for k, v in sorted(counts.items())
        },
    }
    with open(d / "meta.json", "w", encoding="utf-8") as f:
        json.dump(full, f, indent=2, ensure_ascii=False)
    print(
        f"[{name}] rows={full['n_rows']} gold={full['n_gold_labeled']} pos={full['n_pos']} "
        f"base_rate={full['base_rate']:.4f} subtypes={full['subtype_counts']}",
        flush=True,
    )


# --------------------------------------------------------------------------- 20 Newsgroups
def build_20ng():
    from sklearn.datasets import fetch_20newsgroups

    b = fetch_20newsgroups(
        subset="all", remove=("headers", "footers", "quotes"), data_home=str(RAW / "sklearn")
    )
    names = list(b.target_names)
    comp = sorted(n for n in names if n.startswith("comp."))
    texts, y, st, seen = [], [], [], set()
    for t, g in zip(b.data, b.target):
        t = t.strip()
        if len(t) < 20 or t in seen:
            continue
        seen.add(t)
        group = names[g]
        texts.append(t)
        y.append(int(group in comp))
        st.append(comp.index(group) if group in comp else -1)
    save_task(
        "20ng_comp",
        texts,
        y,
        st,
        {
            "source": "sklearn fetch_20newsgroups(subset='all', remove=headers/footers/quotes)",
            "positive_definition": "newsgroup is one of comp.*",
            "subtype_names": comp,
            "has_full_gold": True,
            "notes": "Texts shorter than 20 characters and exact duplicates were removed.",
        },
    )


# --------------------------------------------------------------------------- Toxic Conversations
def build_toxic():
    from huggingface_hub import hf_hub_download

    repo = "mteb/toxic_conversations_50k"
    frames = []
    for split in ("train", "test"):
        p = hf_hub_download(repo, f"{split}.jsonl", repo_type="dataset", cache_dir=str(RAW / "hf"))
        df = pd.read_json(p, lines=True)
        df["split"] = split
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    text_col = next(c for c in ("text", "comment_text", "sentence") if c in df.columns)
    label_col = next(c for c in ("label", "labels") if c in df.columns)
    df["text"] = df[text_col].astype(str).str.strip()
    df = df[df["text"].str.len() >= 5]
    n_before = len(df)
    df = df.drop_duplicates("text", keep="first")
    y = df[label_col].astype(int).to_numpy()
    save_task(
        "toxic_conv",
        df["text"].tolist(),
        y,
        np.where(y == 1, 0, -1),
        {
            "source": f"HuggingFace {repo} (train + test splits merged)",
            "positive_definition": f"{label_col} == 1 (human toxicity annotation)",
            "subtype_names": ["toxic"],
            "has_full_gold": True,
            "notes": (
                f"Columns: {list(df.columns)}. Removed {n_before - len(df)} duplicate texts. "
                "MTEB describes this set as Civil Comments platform comments with toxicity labels."
            ),
        },
    )


# --------------------------------------------------------------------------- Jigsaw
def _read_jigsaw_train() -> pd.DataFrame | None:
    raw = RAW / "jigsaw"
    if (raw / "train.csv").exists():
        return pd.read_csv(raw / "train.csv")
    if (raw / "train.csv.zip").exists():
        return pd.read_csv(raw / "train.csv.zip")
    for z in sorted(raw.glob("*.zip")):
        with zipfile.ZipFile(z) as outer:
            names = outer.namelist()
            if "train.csv" in names:
                with outer.open("train.csv") as f:
                    return pd.read_csv(f)
            if "train.csv.zip" in names:
                inner = zipfile.ZipFile(io.BytesIO(outer.read("train.csv.zip")))
                with inner.open("train.csv") as f:
                    return pd.read_csv(f)
    return None


def build_jigsaw():
    df = _read_jigsaw_train()
    if df is None:
        print(
            "[jigsaw] Missing data. Download train.csv (or the competition zip) of the Kaggle "
            "'Jigsaw Toxic Comment Classification Challenge' and put it in data/raw/jigsaw/.",
            flush=True,
        )
        return False
    texts = df["comment_text"].astype(str).str.strip().tolist()
    threat = df["threat"].to_numpy().astype(int)
    save_task(
        "jigsaw_threat",
        texts,
        threat,
        np.where(threat == 1, 0, -1),
        {
            "source": "Kaggle Jigsaw Toxic Comment Classification Challenge, train.csv",
            "positive_definition": "threat == 1",
            "subtype_names": ["threat"],
            "has_full_gold": True,
        },
    )
    names = ["threat", "identity_hate", "severe_toxic"]
    cols = df[names].to_numpy().astype(int)
    y = cols.max(axis=1)
    first = np.argmax(cols, axis=1)
    save_task(
        "jigsaw_severe",
        None,
        y,
        np.where(y == 1, first, -1),
        {
            "source": "Kaggle Jigsaw Toxic Comment Classification Challenge, train.csv",
            "positive_definition": "threat OR identity_hate OR severe_toxic",
            "subtype_names": names,
            "has_full_gold": True,
            "emb_from": "jigsaw_threat",
            "notes": "Subtype priority when several labels apply: threat > identity_hate > severe_toxic.",
        },
    )
    return True


# --------------------------------------------------------------------------- LTS shark
def build_lts_shark():
    raw = RAW / "lts"
    if not (raw / "shark_trophy.csv").exists() or not (raw / "validation_sharks.csv").exists():
        print(
            "[lts_shark] Missing data/raw/lts/shark_trophy.csv and validation_sharks.csv "
            "(from github.com/VIDA-NYU/LTS, folder data_use_cases).",
            flush=True,
        )
        return False
    from .lts import build as build_lts

    build_lts(raw, save_task)
    return True


BUILDERS = {
    "20ng_comp": build_20ng,
    "toxic_conv": build_toxic,
    "jigsaw": build_jigsaw,
    "lts_shark": build_lts_shark,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", nargs="+", required=True, choices=sorted(BUILDERS))
    args = ap.parse_args()
    for t in args.task:
        BUILDERS[t]()


if __name__ == "__main__":
    main()
