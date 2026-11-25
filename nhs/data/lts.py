"""LTS shark task (Barbosa et al., SIGMOD 2025; github.com/VIDA-NYU/LTS, folder data_use_cases).

shark_trophy.csv       14,533 eBay ads with a `label` column distributed with the dataset (5.5% positive).
                       The LTS code reads it with `-labeling file`. Provenance is not documented; it
                       behaves like an LLM labeling pass (fossils mostly negative, rays positive) with
                       visible noise (e.g. postcards, megalodon fossils labeled positive).
validation_sharks.csv  499 ads with expert labels (`label`; agrees with the partial `relevant` column
                       on every row where both exist). 102 positives (20%), so not a random sample.

Task layout:
  pool rows  = crawled ads; the labeler returns their `label` (saved as labels_lts_label.npy)
  test rows  = expert-labeled ads (eval_mask); gold y only exists there; never sampled
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..paths import DATA


def _clean(title: str) -> str:
    return re.sub(r"\s*\|\s*eBay\s*$", "", str(title)).strip()


def _key(title: str) -> str:
    return " ".join(_clean(title).lower().split())


def build(raw, save_task):
    ads = pd.read_csv(raw / "shark_trophy.csv")
    val = pd.read_csv(raw / "validation_sharks.csv")
    ads = ads[ads["title"].notna() & ads["label"].notna()].copy()
    val = val[val["title"].notna() & val["label"].notna()].copy()
    ads["_key"] = ads["title"].map(_key)
    val["_key"] = val["title"].map(_key)
    val = val.drop_duplicates("_key")
    overlap = ads["_key"].isin(set(val["_key"]))
    pool = ads[~overlap].drop_duplicates("_key")

    texts = pool["title"].map(_clean).tolist() + val["title"].map(_clean).tolist()
    n_pool, n_val = len(pool), len(val)
    y = np.r_[-np.ones(n_pool, dtype=int), val["label"].astype(int).to_numpy()]
    eval_mask = np.r_[np.zeros(n_pool, dtype=bool), np.ones(n_val, dtype=bool)]
    llm = np.r_[pool["label"].astype(int).to_numpy(), -np.ones(n_val, dtype=int)]
    save_task(
        "lts_shark",
        texts,
        y,
        np.where(y == 1, 0, -1),
        {
            "source": "github.com/VIDA-NYU/LTS data_use_cases/shark_trophy.csv + validation_sharks.csv",
            "positive_definition": "product made from a real shark, ray or chimaera "
                                   "(not fossils, single teeth, images or synthetic items)",
            "subtype_names": ["shark product"],
            "has_full_gold": False,
            "label_sources": {"lts_label": "the `label` column shipped with shark_trophy.csv"},
            "notes": (
                f"{n_pool} pool ads ({int(pool['label'].sum())} labeled positive by the shipped labels, "
                f"{pool['label'].mean():.3f}); {n_val} expert-labeled test ads ({int(val['label'].sum())} positive). "
                f"{int(overlap.sum())} crawled ads duplicated a test title and were dropped from the pool. "
                "Texts are ad titles without the trailing '| eBay'."
            ),
        },
        eval_mask=eval_mask,
    )
    np.save(DATA / "lts_shark" / "labels_lts_label.npy", llm.astype(np.int8))
