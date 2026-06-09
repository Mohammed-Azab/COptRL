from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import yaml


def _find_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / 'config' / 'eval' / 'scenarios').is_dir():
            return parent
    raise FileNotFoundError("repo root with config/eval/scenarios not found")


def _load_catalog() -> dict[int, dict]:
    root = _find_root()
    p = root / 'config' / 'road' / 'speed_bumps.json'
    data = json.loads(p.read_text())
    return {b['id']: b for b in data['bumps']}


def list_scenarios() -> list[str]:
    d = _find_root() / 'config' / 'eval' / 'scenarios'
    return sorted(p.stem for p in d.glob('*.yaml'))


def load_scenario(name_or_path: str) -> tuple[list, float, str, str]:
    # loads a named scenario yaml and returns (bumps, speed_m_s, name, description)
    # bumps: list of (x_start_m, height_m, width_m) tuples for RoadGenerator
    # speed_m_s: vehicle speed in m/s (converted from km/h in the yaml)
    path = Path(name_or_path)
    if not path.exists():
        d = _find_root() / 'config' / 'eval' / 'scenarios'
        path = d / f'{name_or_path}.yaml'
    if not path.exists():
        available = list_scenarios()
        raise FileNotFoundError(
            f"Scenario '{name_or_path}' not found. Available: {available}"
        )

    with open(path) as fh:
        cfg = yaml.safe_load(fh)

    catalog = _load_catalog()
    bumps: list[tuple[float, float, float]] = []
    for entry in cfg['bumps']:
        cid  = int(entry['catalog_id'])
        cat  = catalog[cid]
        bumps.append((
            float(entry['x_start_m']),
            float(cat['height_m']),
            float(cat['width_m']),
        ))

    speed_m_s = float(cfg['vehicle_speed_kmh']) / 3.6
    return bumps, speed_m_s, cfg['name'], cfg.get('description', '')


def make_road_generator(name_or_path: str):
    # load a scenario and hand back a ready-to-use RoadGenerator
    from road.road_generator import RoadGenerator

    bumps, speed, _, _ = load_scenario(name_or_path)
    gen = RoadGenerator(profile='speed_bump', vehicle_speed=speed)
    gen._bumps = bumps
    return gen
