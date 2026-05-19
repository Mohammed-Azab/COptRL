"""
Extract 10 road-excitation scenarios from the CARD dataset and save them as
NPZ files under config/scenarios/.

Each NPZ contains only numeric arrays (no pickle):
    arc_m   : arc-length positions starting from 0 (float32, m)
    z_m     : road vertical displacement (float32, m), FL wheel
    v_ref   : reference entry speed (float32 scalar, m/s)

Metadata (source, description) is written to scenarios_index.json.

Usage:
    python scripts/preprocess_card_scenarios.py \
        --card_root /home/ubuntu/myRepo/perception \
        --out_dir   config/scenarios
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_wheel_fl(export_dir: Path):
    p = export_dir / "wheel_excitement.json"
    with open(p) as fh:
        d = json.load(fh)
    fl   = d["wheel_results"]["cariad_wheel_FL"]["wheel_excitement"]
    arc  = np.array([e[2] for e in fl], dtype=np.float64)
    z    = np.array([e[3] for e in fl], dtype=np.float64)
    return arc, z


def _extract_segment(arc_full, z_full, s_start, s_end,
                     v_ref, source, desc, resample_ds=0.01):
    """Slice arc range and resample to uniform ds grid."""
    mask    = (arc_full >= s_start) & (arc_full <= s_end)
    arc_seg = arc_full[mask]
    z_seg   = z_full[mask]

    if len(arc_seg) < 4:
        raise ValueError(f"Too few points in segment [{s_start}, {s_end}] m")

    arc_local = arc_seg - arc_seg[0]
    arc_out   = np.arange(0.0, arc_local[-1] + resample_ds, resample_ds)
    fn        = interp1d(arc_local, z_seg, kind="linear",
                         bounds_error=False, fill_value=(z_seg[0], z_seg[-1]))
    z_out     = fn(arc_out)

    return {
        "arc_m":   arc_out.astype(np.float32),
        "z_m":     z_out.astype(np.float32),
        "v_ref":   float(v_ref),
        "source":  source,
        "desc":    desc,
    }


# ---------------------------------------------------------------------------
# scenario catalogue
# ---------------------------------------------------------------------------
# bump in Nardo_speedbump1 at arc ≈ 36.3 m; total arc ≈ 83.35 m; ~7 m/s
# roughest 50 m in export/ at arc 725-775 m (z_std ≈ 2.1 cm)

def _build_catalogue(card_root: Path):
    sb1   = _load_wheel_fl(card_root / "output/italy/Nardo/Nardo_speedbump1/export")
    long_ = _load_wheel_fl(card_root / "export/export")
    a1, z1 = sb1
    a2, z2 = long_

    return [
        _extract_segment(a1, z1, 0.0, 83.0, 7.0,
                         "Nardo_speedbump1",
                         "Full speed-bump run at real entry speed (7 m/s)."),
        _extract_segment(a1, z1, 0.0, 83.0, 5.0,
                         "Nardo_speedbump1",
                         "Full speed-bump run replayed at reduced speed (5 m/s)."),
        _extract_segment(a1, z1, 0.0, 83.0, 3.0,
                         "Nardo_speedbump1",
                         "Full speed-bump run replayed at creep speed (3 m/s)."),
        _extract_segment(a1, z1, 0.0, 30.0, 7.0,
                         "Nardo_speedbump1",
                         "Pre-bump flat approach only (s=0-30 m, v=7 m/s)."),
        _extract_segment(a2, z2, 0.0, 200.0, 10.0,
                         "export",
                         "Early highway section, smooth-ish (s=0-200 m, v=10 m/s)."),
        _extract_segment(a2, z2, 700.0, 800.0, 12.0,
                         "export",
                         "Roughest 100 m segment at high speed (s=700-800 m, v=12 m/s)."),
        _extract_segment(a2, z2, 550.0, 650.0, 9.0,
                         "export",
                         "Second-roughest segment at cruise speed (s=550-650 m, v=9 m/s)."),
        _extract_segment(a2, z2, 1125.0, 1225.0, 14.0,
                         "export",
                         "High-speed rough section (s=1125-1225 m, v=14 m/s)."),
        _extract_segment(a2, z2, 0.0, 150.0, 6.0,
                         "export",
                         "Early highway at low speed (s=0-150 m, v=6 m/s)."),
        _extract_segment(a2, z2, 300.0, 600.0, 10.0,
                         "export",
                         "Long mixed-rough section (s=300-600 m, v=10 m/s)."),
    ]


NAMES = [
    "sb1_v7_full",
    "sb1_v5_full",
    "sb1_v3_creep",
    "sb1_v7_approach",
    "exp_v10_early",
    "exp_v12_rough",
    "exp_v9_rough2",
    "exp_v14_fast",
    "exp_v6_slow",
    "exp_v10_long",
]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card_root", default="/home/ubuntu/myRepo/perception")
    ap.add_argument("--out_dir",   default="config/scenarios")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    catalogue = _build_catalogue(Path(args.card_root))

    index = {}
    for name, sc in zip(NAMES, catalogue):
        out_path = out_dir / f"{name}.npz"
        # Save numeric arrays only — no object arrays, no serialization issues
        np.savez(out_path,
                 arc_m=sc["arc_m"],
                 z_m=sc["z_m"],
                 v_ref=np.float32(sc["v_ref"]))

        arc, z = sc["arc_m"], sc["z_m"]
        index[name] = {
            "file":      f"{name}.npz",
            "source":    sc["source"],
            "desc":      sc["desc"],
            "v_ref":     sc["v_ref"],
            "arc_len_m": round(float(arc[-1]), 2),
            "n_pts":     int(len(arc)),
            "z_std_cm":  round(float(z.std() * 100), 3),
            "z_max_cm":  round(float(np.max(np.abs(z)) * 100), 1),
        }
        print(
            f"[{name:25s}]  {len(arc):5d} pts  "
            f"{arc[-1]:.0f} m  v={sc['v_ref']:.0f} m/s  "
            f"z_std={z.std()*100:.2f} cm  max|z|={float(np.max(np.abs(z)))*100:.1f} cm"
            f"  → {out_path}"
        )

    idx_path = out_dir / "scenarios_index.json"
    with open(idx_path, "w") as fh:
        json.dump(index, fh, indent=2)

    print(f"\n{len(catalogue)} scenarios saved to {out_dir}/")
    print(f"Metadata index → {idx_path}")


if __name__ == "__main__":
    main()
