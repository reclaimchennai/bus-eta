#!/usr/bin/env python3
"""Async Local Open-Weight LLM Context Sidecar for Unstructured Indian Traffic Disruptions.

Why an async sidecar (instead of calling an LLM in the 10ms per-bus ETA loop)?
  - Indian bus delays are disproportionately hit by unscheduled events invisible to GPS history:
    waterlogging, VIP convoys, rallies/bandhs, temple/festival processions, metro barricades,
    breakdowns, and Traffic Police diversions (e.g. Delhi dtp_closures.py, Chennai/Bengaluru/Mumbai X feeds).
  - Running a small local open-weight LLM (Qwen2.5-7B / Llama-3.1-8B via Ollama/MLX/vLLM) every
    5-15 minutes parses unstructured multilingual alerts (English/Hinglish/Tamil/Kannada/Marathi)
    into structured corridor risk multipliers saved to a fast JSON/Redis lookup table consumed
    in <0.1ms by our LightGBM/CatBoost ETA engine!

Supports:
  - Local Ollama HTTP API (http://localhost:11434/api/generate)
  - Deterministic Indian-transit regex fallback when no local LLM server is running.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import requests

SYSTEM_PROMPT = """You are an Indian urban transit traffic analyst.
Extract structured bus-delay risk from a traffic/weather advisory.
Return ONLY valid JSON with keys:
  - event_type: one of ["waterlogging", "road_closure", "vip_movement", "rally_procession", "construction", "breakdown_accident", "festival", "other"]
  - severity: float in [0.0, 1.0] (0.2=minor slowdown, 0.5=moderate jam, 0.8=severe bottleneck, 1.0=full closure/diversion)
  - speed_multiplier: float in [0.25, 1.0] (expected bus speed fraction vs normal, e.g. 0.55 means 45% slower)
  - duration_hours: float (expected duration from posting)
  - landmarks: list of short location/road/junction strings mentioned (e.g. ["Silk Board", "ITO", "Anna Salai"])
  - bus_diversion_likely: bool
"""

FALLBACK_RULES: list[tuple[str, str, float, float, float]] = [
    (r"water\s*log|flood|inundat|submerg|heavy rain|cyclone", "waterlogging", 0.80, 0.45, 6.0),
    (r"closed|closure|block(?:ed|age)|no\s+entry|barricad|divert|diversion", "road_closure", 0.85, 0.40, 8.0),
    (r"vip|vvip|convoy|pm\s+visit|president|minister", "vip_movement", 0.70, 0.50, 3.0),
    (r"rally|protest|bandh|march|procession|yatra|juloos|temple\s+car|chariot", "rally_procession", 0.75, 0.45, 5.0),
    (r"metro\s+work|flyover\s+work|pothole|repair|excavat|pipeline|caving", "construction", 0.55, 0.65, 24.0),
    (r"breakdown|accident|collision|overturn|bus\s+stuck", "breakdown_accident", 0.60, 0.60, 2.0),
]


def parse_with_ollama(text: str, model: str = "qwen2.5:7b", url: str = "http://localhost:11434/api/generate") -> dict[str, Any] | None:
    try:
        resp = requests.post(
            url,
            json={
                "model": model,
                "prompt": f"{SYSTEM_PROMPT}\n\nAdvisory:\n{text}\n\nJSON:",
                "format": "json",
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 256},
            },
            timeout=15,
        )
        if resp.status_code != 200:
            return None
        raw = resp.json().get("response", "").strip()
        data = json.loads(raw)
        return {
            "event_type": str(data.get("event_type", "other")),
            "severity": float(max(0.0, min(1.0, data.get("severity", 0.5)))),
            "speed_multiplier": float(max(0.25, min(1.0, data.get("speed_multiplier", 0.7)))),
            "duration_hours": float(max(0.5, min(72.0, data.get("duration_hours", 4.0)))),
            "landmarks": [str(x) for x in (data.get("landmarks") or [])][:10],
            "bus_diversion_likely": bool(data.get("bus_diversion_likely", False)),
            "parser": f"ollama:{model}",
        }
    except Exception:
        return None


def parse_with_rules(text: str) -> dict[str, Any]:
    low = text.lower()
    for pat, etype, sev, mult, dur in FALLBACK_RULES:
        if re.search(pat, low):
            # Extract capitalized place/road phrases as rough landmarks
            lms = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", text)
            lms = [x for x in dict.fromkeys(lms) if x.lower() not in ("traffic", "alert", "advisory", "police", "today", "due")][:8]
            return {
                "event_type": etype,
                "severity": sev,
                "speed_multiplier": mult,
                "duration_hours": dur,
                "landmarks": lms,
                "bus_diversion_likely": etype in ("road_closure", "waterlogging", "rally_procession"),
                "parser": "regex_fallback",
            }
    return {
        "event_type": "other",
        "severity": 0.30,
        "speed_multiplier": 0.85,
        "duration_hours": 3.0,
        "landmarks": [],
        "bus_diversion_likely": False,
        "parser": "regex_fallback",
    }


def extract_disruption(text: str, use_ollama: bool = True, model: str = "qwen2.5:7b") -> dict[str, Any]:
    if use_ollama:
        res = parse_with_ollama(text, model=model)
        if res is not None:
            return res
    return parse_with_rules(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-jsonl", help="Path to alerts JSONL (e.g. dtp_closures items or X feed)")
    ap.add_argument("--out-json", default="~/projects/ETA/data/processed/disruptions_live.json")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--no-ollama", action="store_true")
    a = ap.parse_args()

    samples = [
        "Traffic Alert: Heavy waterlogging reported near Silk Board Junction and Outer Ring Road towards Marathahalli. Slow movement expected for next 4 hours.",
        "Delhi Traffic Police Advisory: Traffic is affected on Ring Road from Ashram towards ITO due to breakdown of a DTC bus near Sarai Kale Khan.",
        "Chennai Traffic: Due to Metro Rail girder launching on Anna Salai near Nandanam, MTC buses are diverted via Chamiers Road from 10 PM to 5 AM.",
    ]
    if a.input_jsonl and Path(a.input_jsonl).exists():
        items = [json.loads(ln) for ln in Path(a.input_jsonl).read_text().splitlines() if ln.strip()]
        texts = [f"{x.get('title', '')}. {x.get('text', '')}".strip() for x in items]
    else:
        texts = samples

    out = []
    for txt in texts:
        parsed = extract_disruption(txt, use_ollama=not a.no_ollama, model=a.model)
        parsed["raw_text"] = txt
        parsed["ts"] = int(time.time())
        out.append(parsed)
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(out, indent=2))
    print(f"Parsed {len(out)} advisories -> {a.out_json}")
    for o in out[:3]:
        print(f"  [{o['parser']}] {o['event_type']} sev={o['severity']} mult={o['speed_multiplier']} lms={o['landmarks']}")


if __name__ == "__main__":
    main()
