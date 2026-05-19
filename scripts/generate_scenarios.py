"""
Road scenario generator — reads a YAML config and saves each scenario as a
pandas-backed JSON file.

OUTPUT FORMAT
─────────────
Each scenario → config/scenarios/<name>.json :

    {
      "v_ref": 7.0,          ← reference entry speed (m/s)
      "arc_m": [0.0, 0.01, 0.02, ...],   ← arc-length positions (m)
      "z_m":   [0.0023, -0.006, ...]      ← road vertical displacement (m)
    }

Loaded with pandas in one line:
    import pandas as pd, json
    meta = json.load(open("sb1_v7_full.json"))
    df   = pd.DataFrame({"arc_m": meta["arc_m"], "z_m": meta["z_m"]})

Companion index → config/scenarios/scenarios_index.json :
    Maps each scenario name to its metadata (v_ref, source, stats, …).

INPUT CONFIG FORMAT  (YAML)
───────────────────────────
card_root: /home/ubuntu/myRepo/perception   # CARD data root (absolute or relative)
out_dir:   config/scenarios                 # where to write output files
resample_ds: 0.01                           # arc-length grid step (metres)

scenarios:
  - name:       sb1_v7_full                 # output file stem (no extension)
    wheel_file: output/italy/Nardo/Nardo_speedbump1/export/wheel_excitement.json
                                            # path relative to card_root
    wheel:      FL                          # FL | FR | RL | RR  (default: FL)
    s_start:    0.0                         # arc start (m)
    s_end:      83.0                        # arc end (m)
    v_ref:      7.0                         # reference speed (m/s)
    desc:       "Full speed-bump run"       # optional description

  - name: custom_segment
    wheel_file: export/export/wheel_excitement.json
    wheel: FR
    s_start: 700.0
    s_end:   800.0
    v_ref:   12.0
    desc: "Roughest 100 m at high speed"

USAGE
─────
    # Use default built-in config:
    python scripts/generate_scenarios.py

    # Supply your own YAML:
    python scripts/generate_scenarios.py --config path/to/my_scenarios.yaml

    # Override output dir:
    python scripts/generate_scenarios.py --out_dir /tmp/my_scenarios
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.interpolate import interp1d


# ─────────────────────────────────────────────────────────────── wheel keys ──

_WHEEL_KEY = {
    "FL": "cariad_wheel_FL",
    "FR": "cariad_wheel_FR",
    "RL": "cariad_wheel_RL",
    "RR": "cariad_wheel_RR",
}


# ──────────────────────────────────────────────────────────────── loaders ────

def _load_wheel(wheel_file: Path, wheel: str = "FL"):
    """
    Load arc-length and vertical-displacement arrays from wheel_excitement.json.

    Returns
    -------
    arc_m : ndarray  arc-length values (m), monotonically increasing
    z_m   : ndarray  vertical displacement of tyre contact point (m)
    """
    key = _WHEEL_KEY.get(wheel.upper())
    if key is None:
        raise ValueError(f"Invalid wheel '{wheel}'. Choose from: FL, FR, RL, RR")

    with open(wheel_file) as fh:
        d = json.load(fh)

    entries = d["wheel_results"][key]["wheel_excitement"]
    arc = np.array([e[2] for e in entries], dtype=np.float64)
    z   = np.array([e[3] for e in entries], dtype=np.float64)
    return arc, z


def _extract_segment(arc_full: np.ndarray, z_full: np.ndarray,
                     s_start: float, s_end: float,
                     resample_ds: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """
    Slice [s_start, s_end] and resample to a uniform ds grid.

    Returns
    -------
    arc_out : ndarray  arc positions starting from 0 (m)
    z_out   : ndarray  road displacement (m)
    """
    mask    = (arc_full >= s_start) & (arc_full <= s_end)
    arc_seg = arc_full[mask]
    z_seg   = z_full[mask]

    if len(arc_seg) < 4:
        raise ValueError(
            f"Only {len(arc_seg)} points in [{s_start}, {s_end}] m. "
            "Check s_start/s_end against the actual arc range of the file."
        )

    arc_local = arc_seg - arc_seg[0]
    arc_out   = np.arange(0.0, arc_local[-1] + resample_ds, resample_ds)
    fn = interp1d(arc_local, z_seg, kind="linear",
                  bounds_error=False,
                  fill_value=(float(z_seg[0]), float(z_seg[-1])))
    z_out = fn(arc_out)
    return arc_out.astype(np.float32), z_out.astype(np.float32)


# ──────────────────────────────────────────────────────────────── saving ─────

def _save_scenario(out_path: Path, arc_m: np.ndarray, z_m: np.ndarray,
                   v_ref: float) -> None:
    """
    Save one scenario as a JSON file.

    File structure
    ──────────────
    {
      "v_ref": <float>,           reference entry speed (m/s)
      "arc_m": [0.0, 0.01, ...], arc-length positions (m)
      "z_m":   [...],             road vertical displacement (m)
    }

    Load with:
        meta = json.load(open(path))
        df   = pd.DataFrame({"arc_m": meta["arc_m"], "z_m": meta["z_m"]})
    """
    payload = {
        "v_ref": round(float(v_ref), 4),
        "arc_m": [round(float(x), 4) for x in arc_m],
        "z_m":   [round(float(z), 6) for z in z_m],
    }
    with open(out_path, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"))   # compact, no extra whitespace


# ──────────────────────────────────────────────────────────── default cfg ────

_DEFAULT_CONFIG = {
    "card_root":   "/home/ubuntu/myRepo/perception",
    "out_dir":     "config/scenarios",
    "resample_ds": 0.01,
    "scenarios": [
        # ── Nardo speed-bump 1 (arc 0–83 m, bump at ~36 m, real speed ~7 m/s) ──
        {"name": "sb1_v7_full",      "wheel_file": "output/italy/Nardo/Nardo_speedbump1/export/wheel_excitement.json",
         "wheel": "FL", "s_start": 0.0,  "s_end": 83.0,  "v_ref": 7.0,
         "desc": "Full speed-bump run at real entry speed (7 m/s)."},

        {"name": "sb1_v5_full",      "wheel_file": "output/italy/Nardo/Nardo_speedbump1/export/wheel_excitement.json",
         "wheel": "FL", "s_start": 0.0,  "s_end": 83.0,  "v_ref": 5.0,
         "desc": "Full speed-bump run replayed at reduced speed (5 m/s)."},

        {"name": "sb1_v3_creep",     "wheel_file": "output/italy/Nardo/Nardo_speedbump1/export/wheel_excitement.json",
         "wheel": "FL", "s_start": 0.0,  "s_end": 83.0,  "v_ref": 3.0,
         "desc": "Full speed-bump run replayed at creep speed (3 m/s)."},

        {"name": "sb1_v7_approach",  "wheel_file": "output/italy/Nardo/Nardo_speedbump1/export/wheel_excitement.json",
         "wheel": "FL", "s_start": 0.0,  "s_end": 30.0,  "v_ref": 7.0,
         "desc": "Pre-bump flat approach only (s=0-30 m, v=7 m/s)."},

        # ── long export/ sequence (arc 0–1927 m, mean speed ~9 m/s) ──
        {"name": "exp_v10_early",    "wheel_file": "export/export/wheel_excitement.json",
         "wheel": "FL", "s_start": 0.0,    "s_end": 200.0,  "v_ref": 10.0,
         "desc": "Early highway, smooth (s=0-200 m, v=10 m/s)."},

        {"name": "exp_v12_rough",    "wheel_file": "export/export/wheel_excitement.json",
         "wheel": "FL", "s_start": 700.0,  "s_end": 800.0,  "v_ref": 12.0,
         "desc": "Roughest 100 m at high speed (s=700-800 m, v=12 m/s)."},

        {"name": "exp_v9_rough2",    "wheel_file": "export/export/wheel_excitement.json",
         "wheel": "FL", "s_start": 550.0,  "s_end": 650.0,  "v_ref": 9.0,
         "desc": "Second-roughest segment (s=550-650 m, v=9 m/s)."},

        {"name": "exp_v14_fast",     "wheel_file": "export/export/wheel_excitement.json",
         "wheel": "FL", "s_start": 1125.0, "s_end": 1225.0, "v_ref": 14.0,
         "desc": "High-speed rough section (s=1125-1225 m, v=14 m/s)."},

        {"name": "exp_v6_slow",      "wheel_file": "export/export/wheel_excitement.json",
         "wheel": "FL", "s_start": 0.0,    "s_end": 150.0,  "v_ref": 6.0,
         "desc": "Early highway at low speed (s=0-150 m, v=6 m/s)."},

        {"name": "exp_v10_long",     "wheel_file": "export/export/wheel_excitement.json",
         "wheel": "FL", "s_start": 300.0,  "s_end": 600.0,  "v_ref": 10.0,
         "desc": "Long mixed-rough section (s=300-600 m, v=10 m/s)."},
    ],
}


# ──────────────────────────────────────────────────────────────────── main ───

def run(cfg: dict) -> None:
    card_root   = Path(cfg["card_root"])
    out_dir     = Path(cfg.get("out_dir", "config/scenarios"))
    resample_ds = float(cfg.get("resample_ds", 0.01))

    out_dir.mkdir(parents=True, exist_ok=True)

    # Cache loaded wheel files to avoid re-reading the same file multiple times
    _wheel_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    index: dict[str, dict] = {}
    ok = errors = 0

    for sc in cfg["scenarios"]:
        name = sc["name"]
        try:
            wf_path = card_root / sc["wheel_file"]
            cache_key = f"{wf_path}::{sc.get('wheel', 'FL')}"

            if cache_key not in _wheel_cache:
                _wheel_cache[cache_key] = _load_wheel(wf_path, sc.get("wheel", "FL"))
            arc_full, z_full = _wheel_cache[cache_key]

            s_start = float(sc["s_start"])
            s_end   = float(sc["s_end"])
            v_ref   = float(sc["v_ref"])

            # Validate arc range
            if s_end > arc_full[-1] + 0.5:
                raise ValueError(
                    f"s_end={s_end} m exceeds file arc range "
                    f"[{arc_full[0]:.1f}, {arc_full[-1]:.1f}] m"
                )

            arc_m, z_m = _extract_segment(arc_full, z_full, s_start, s_end, resample_ds)

            out_path = out_dir / f"{name}.json"
            _save_scenario(out_path, arc_m, z_m, v_ref)

            # Build index entry
            index[name] = {
                "file":      f"{name}.json",
                "source":    sc.get("wheel_file", ""),
                "wheel":     sc.get("wheel", "FL"),
                "desc":      sc.get("desc", ""),
                "v_ref":     v_ref,
                "s_start":   s_start,
                "s_end":     s_end,
                "arc_len_m": round(float(arc_m[-1]), 3),
                "n_pts":     int(len(arc_m)),
                "z_std_cm":  round(float(z_m.std() * 100), 3),
                "z_max_cm":  round(float(np.max(np.abs(z_m)) * 100), 1),
            }

            print(
                f"  ✓  {name:25s}  "
                f"{len(arc_m):6d} pts  {arc_m[-1]:.0f} m  "
                f"v={v_ref:.0f} m/s  "
                f"z_std={z_m.std()*100:.2f} cm  "
                f"max|z|={float(np.max(np.abs(z_m)))*100:.1f} cm"
            )
            ok += 1

        except Exception as exc:
            print(f"  ✗  {name:25s}  ERROR: {exc}")
            errors += 1

    # Write index
    idx_path = out_dir / "scenarios_index.json"
    with open(idx_path, "w") as fh:
        json.dump(index, fh, indent=2)

    print(f"\n{ok} scenarios saved  |  {errors} errors  →  {out_dir}/")
    print(f"Index → {idx_path}")

    if errors:
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate road-excitation scenario JSON files from CARD data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--config", "-c",
        default=None,
        help="Path to a YAML scenario config file. "
             "If omitted, the built-in 10-scenario config is used.",
    )
    ap.add_argument(
        "--out_dir",
        default=None,
        help="Override the output directory from the config.",
    )
    ap.add_argument(
        "--card_root",
        default=None,
        help="Override card_root from the config.",
    )
    ap.add_argument(
        "--dump_config",
        action="store_true",
        help="Print the default YAML config to stdout and exit (use as a template).",
    )
    args = ap.parse_args()

    if args.dump_config:
        print(yaml.dump(_DEFAULT_CONFIG, default_flow_style=False, sort_keys=False))
        return

    if args.config:
        with open(args.config) as fh:
            cfg = yaml.safe_load(fh)
    else:
        cfg = dict(_DEFAULT_CONFIG)

    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    if args.card_root:
        cfg["card_root"] = args.card_root

    run(cfg)


if __name__ == "__main__":
    main()
