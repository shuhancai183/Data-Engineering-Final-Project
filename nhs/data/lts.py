"""LTS shark task (Barbosa et al., SIGMOD 2025; github.com/VIDA-NYU/LTS, folder data_use_cases).

shark_trophy.csv     all crawled ads (no labels)  -> sampling pool
validation_sharks.csv ~500 expert-labeled ads      -> evaluation only (never sampled)

Only ~500 rows carry human labels, so samplers on this task buy labels from a cached real-LLM
labeling pass (nhs.llm_label); F1 is measured on the expert rows.
Column names are detected; if detection fails the available columns are printed.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

TEXT_COLS = ("title", "text", "description", "name", "product", "ad")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def _text_col(df: pd.DataFrame) -> str:
    for c in TEXT_COLS:
        for col in df.columns:
            if col.lower() == c:
                return col
    raise KeyError(f"no text column found; columns: {list(df.columns)}")


def _label_col(df: pd.DataFrame) -> str:
    cands = [c for c in df.columns if "label" in c.lower() or c.lower() in ("relevant", "gold", "y", "class")]
    if not cands:
        raise KeyError(f"no label column found; columns: {list(df.columns)}")
    return cands[0]


def _to01(v) -> int:
    if isinstance(v, (int, np.integer, float, np.floating)) and not pd.isna(v):
        return int(v > 0)
    s = _norm(v)
    if s.startswith("not") or s in ("0", "false", "no", "irrelevant"):
        return 0
    if "relevant" in s or s in ("1", "true", "yes"):
        return 1
    return -1


def build(raw, save_task):
    ads = pd.read_csv(raw / "shark_trophy.csv")
    val = pd.read_csv(raw / "validation_sharks.csv")
    tcol, vtcol, lcol = _text_col(ads), _text_col(val), _label_col(val)
    print(f"[lts_shark] ads columns={list(ads.columns)} | validation columns={list(val.columns)}", flush=True)
    print(f"[lts_shark] text column '{tcol}', validation text '{vtcol}', label '{lcol}'", flush=True)

    ads = ads[ads[tcol].notna()].copy()
    ads["_key"] = ads[tcol].map(_norm)
    ads = ads.drop_duplicates("_key")
    val = val[val[vtcol].notna()].copy()
    val["_key"] = val[vtcol].map(_norm)
    val["_y"] = val[lcol].map(_to01)
    val = val[val["_y"] >= 0].drop_duplicates("_key")

    pool = ads[~ads["_key"].isin(set(val["_key"]))]
    texts = pool[tcol].astype(str).str.strip().tolist() + val[vtcol].astype(str).str.strip().tolist()
    y = np.r_[-np.ones(len(pool), dtype=int), val["_y"].to_numpy()]
    eval_mask = np.r_[np.zeros(len(pool), dtype=bool), np.ones(len(val), dtype=bool)]
    save_task(
        "lts_shark",
        texts,
        y,
        np.where(y == 1, 0, -1),
        {
            "source": "github.com/VIDA-NYU/LTS data_use_cases/shark_trophy.csv + validation_sharks.csv",
            "positive_definition": "expert label: product made from a real shark, ray or chimaera "
                                   "(not fossils, single teeth, images or synthetic items)",
            "subtype_names": ["shark product"],
            "has_full_gold": False,
            "notes": (
                f"{len(ads)} unique ads, {len(val)} expert-labeled ads "
                f"({int(ads['_key'].isin(set(val['_key'])).sum())} of them also in the crawl). "
                "Expert rows are reserved for evaluation and removed from the sampling pool."
            ),
        },
        eval_mask=eval_mask,
    )
