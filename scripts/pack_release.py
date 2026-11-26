#!/usr/bin/env python3
"""Pack data/ into GitHub Release assets (each under 2 GiB) plus data/MANIFEST.json.

    python scripts/pack_release.py OUT_DIR            # build the archives
    gh release upload data-v1 OUT_DIR/*               # (scripts/publish_data.sh does this)

Each group of data/ becomes one .tar.gz with the files exactly as they sit in data/, split into parts of PART_MB
when larger (concatenate the parts, then untar). MANIFEST.json lists every asset with its size and sha256 and the
folder it unpacks into; scripts/fetch_data.sh downloads, verifies and unpacks them.
"""
import hashlib
import json
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PART_MB = 1900
TAG = "data-v1"

# asset name -> folder under data/ (each sub-folder of raw/ and rtc_src/ is its own asset)
def groups():
    g = {}
    for parent in ("raw", "rtc_src"):
        for d in sorted((DATA / parent).iterdir()):
            if d.is_dir():
                g[f"{parent}-{d.name}"] = f"{parent}/{d.name}"
    for d in ("processed", "v3_cache", "rtc_cache", "meta", "audit", "shadow", "shadow_rtc", "btp"):
        if (DATA / d).is_dir():
            g[d] = d
    return g


SKIP = {".DS_Store", "cfg.json", "feed_local.json"}   # cfg.json / feed_local.json hold our ssh hosts




def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"tag": TAG, "built": time.strftime("%Y-%m-%d"), "note": "concatenate an asset's parts in order and untar (gzip) into data/<into>", "assets": []}
    for name, rel in groups().items():
        src = DATA / rel
        files = sorted(p for p in src.rglob("*") if p.is_file() and p.name not in SKIP)
        if not files:
            continue
        tar = out / f"{name}.tar.gz"
        done = sorted(out.glob(f"{name}.tar.gz.part*"))
        if done and not tar.exists():            # built and split on an earlier run
            manifest["assets"].append({"name": name, "into": rel,
                                       "files": [{"asset": p.name, "bytes": p.stat().st_size, "sha256": sha(p)} for p in done]})
            continue
        if not tar.exists():
            t0 = time.time()
            with tarfile.open(tar, "w:gz", compresslevel=6, format=tarfile.PAX_FORMAT) as tf:
                for p in files:
                    tf.add(p, str(p.relative_to(src)), recursive=False)
            print(f"{name}: {len(files)} files, {tar.stat().st_size / 1e6:.0f} MB, {time.time() - t0:.0f} s", flush=True)
        parts = [tar]
        if tar.stat().st_size > PART_MB * 1024 * 1024:
            parts = []
            with open(tar, "rb") as f:
                i = 0
                while True:
                    chunk = f.read(PART_MB * 1024 * 1024)
                    if not chunk:
                        break
                    pp = out / f"{name}.tar.gz.part{i:02d}"
                    pp.write_bytes(chunk)
                    parts.append(pp)
                    i += 1
            tar.unlink()
        manifest["assets"].append({"name": name, "into": rel,
                                   "files": [{"asset": p.name, "bytes": p.stat().st_size, "sha256": sha(p)} for p in parts]})
    (DATA / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print("wrote data/MANIFEST.json:", sum(len(a["files"]) for a in manifest["assets"]), "assets")


if __name__ == "__main__":
    main()
