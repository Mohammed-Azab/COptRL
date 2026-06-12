#!/usr/bin/env python3
"""
Generate pre-computed single-bump training scenarios for the COptRL curriculum.
Saves config/train/scenarios/{easy,medium,hard,expert}/scenario_NNN.yaml.

Each YAML contains: catalog_id, catalog_name, speed_kmh, flat_start_m,
difficulty, difficulty_score, height_m, width_m.

Usage:
  python scripts/generate_train_scenarios.py
  python scripts/generate_train_scenarios.py --per-level 100
  python scripts/generate_train_scenarios.py --flat-starts 30 60 90
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import yaml

_ROOT         = Path(__file__).resolve().parents[1]
_CATALOG_PATH = _ROOT / "config" / "road" / "speed_bumps.json"
_DEFAULT_OUT  = _ROOT / "config" / "train" / "scenarios"

# Intrinsic difficulty rank per bump (0=easiest, 1=hardest), from Mandl 2021:
#   3 long_bump        H=12.5cm W=9.5m  heave-resonance    → 0.00
#   4 raised_crosswalk H=10cm   W=5.0m  wide crosswalk     → 0.25
#   1 medium_bump      H=6.25cm W=2.22m human-sensitivity  → 0.50
#   0 short_bump       H=2.5cm  W=0.92m wheel-hop          → 0.75
#   2 severe_bump      H=10cm   W=1.0m  narrow+tall        → 1.00
_BUMP_RANK: dict[int, float] = {3: 0.00, 4: 0.25, 1: 0.50, 0: 0.75, 2: 1.00}

_V_MIN_KMH = 18.0
_V_MAX_KMH = 50.0

_THRESHOLDS = [
    ("easy",   0.00, 0.30),
    ("medium", 0.30, 0.57),
    ("hard",   0.57, 0.78),
    ("expert", 0.78, 1.01),
]


def _load_catalog() -> list[dict]:
    return json.loads(_CATALOG_PATH.read_text())["bumps"]


def _difficulty_score(catalog_id: int, speed_kmh: float) -> float:
    bump_r  = _BUMP_RANK[catalog_id]
    speed_n = float(np.clip((speed_kmh - _V_MIN_KMH) / (_V_MAX_KMH - _V_MIN_KMH), 0.0, 1.0))
    return 0.6 * bump_r + 0.4 * speed_n


def _assign_difficulty(score: float) -> str:
    for name, lo, hi in _THRESHOLDS:
        if lo <= score < hi:
            return name
    return "expert"


def generate(
    per_level: int = 200,
    flat_starts: list[float] | None = None,
    out_dir: Path = _DEFAULT_OUT,
    seed: int = 42,
) -> dict[str, int]:
    """Generate exactly per_level scenarios per difficulty category."""
    if flat_starts is None:
        flat_starts = [30.0, 45.0, 60.0, 75.0]

    catalog     = _load_catalog()
    catalog_ids = [b["id"] for b in catalog]

    # Large pool so every difficulty bucket has ≥ per_level unique entries
    n_speeds = max(100, per_level * 2)
    speeds   = np.linspace(_V_MIN_KMH, _V_MAX_KMH, n_speeds)

    by_diff: dict[str, list[dict]] = {d: [] for d, *_ in _THRESHOLDS}
    for cid in catalog_ids:
        bump = catalog[cid]
        for v_raw in speeds:
            v = round(float(v_raw), 2)
            for fs in flat_starts:
                score = _difficulty_score(cid, v)
                diff  = _assign_difficulty(score)
                by_diff[diff].append({
                    "catalog_id":        cid,
                    "catalog_name":      bump["name"],
                    "speed_kmh":         v,
                    "flat_start_m":      float(fs),
                    "difficulty":        diff,
                    "difficulty_score":  round(score, 4),
                    "height_m":          bump["height_m"],
                    "width_m":           bump["width_m"],
                })

    rng = np.random.default_rng(seed)
    selected: list[dict] = []
    for diff in ("easy", "medium", "hard", "expert"):
        items = by_diff[diff]
        if len(items) < per_level:
            print(f"  WARNING: only {len(items)} unique {diff} scenarios; upsampling to {per_level}")
            idxs = rng.choice(len(items), per_level, replace=True)
        else:
            idxs = rng.choice(len(items), per_level, replace=False)
        selected.extend([items[int(i)] for i in idxs])

    _ord = {"easy": 0, "medium": 1, "hard": 2, "expert": 3}
    selected.sort(key=lambda c: (_ord[c["difficulty"]], c["difficulty_score"]))

    for diff in ("easy", "medium", "hard", "expert"):
        p = out_dir / diff
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)

    counters: dict[str, int] = {"easy": 0, "medium": 0, "hard": 0, "expert": 0}
    for sc in selected:
        diff = sc["difficulty"]
        counters[diff] += 1
        fname = out_dir / diff / f"scenario_{counters[diff]:03d}.yaml"
        fname.write_text(yaml.dump(sc, default_flow_style=False, sort_keys=False))

    manifest = {
        "total":                 sum(counters.values()),
        "per_difficulty":        dict(counters),
        "catalog_source":        "Mandl 2021 (speed_bumps.json)",
        "v_range_kmh":           [_V_MIN_KMH, _V_MAX_KMH],
        "flat_start_m_values":   flat_starts,
        "n_speeds_pool":         n_speeds,
        "per_level_requested":   per_level,
        "difficulty_thresholds": {
            name: {"score_min": lo, "score_max": hi}
            for name, lo, hi in _THRESHOLDS
        },
        "bump_rank": _BUMP_RANK,
    }
    (out_dir / "manifest.yaml").write_text(
        yaml.dump(manifest, default_flow_style=False, sort_keys=False)
    )
    return counters


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate single-bump training scenarios for COptRL curriculum.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--per-level", type=int, default=200,
                   help="Scenarios per difficulty level")
    p.add_argument("--flat-starts", type=float, nargs="+",
                   default=[30.0, 45.0, 60.0, 75.0],
                   help="Flat-start distances in metres")
    p.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT,
                   help="Output directory")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(f"Generating {args.per_level} scenarios per level → {args.out_dir}")
    counts = generate(
        per_level=args.per_level,
        flat_starts=args.flat_starts,
        out_dir=args.out_dir,
        seed=args.seed,
    )
    total = sum(counts.values())
    print(f"Generated {total} scenarios:")
    for diff in ("easy", "medium", "hard", "expert"):
        print(f"  {diff:8s}: {counts.get(diff, 0):3d}")
    print(f"Manifest → {args.out_dir / 'manifest.yaml'}")


if __name__ == "__main__":
    main()
