#!/usr/bin/env python3
"""
Convert a file containing multiple JSON records (array, NDJSON, or concatenated JSON)
into a normalized CSV of landmark vectors.

Each output row:
 label, x0, y0, x1, y1, ..., x20, y20

Usage examples:
python scripts/convert_jsonlist_to_csv.py --infile json_palm.txt --out data/palm_embeddings.csv --label open_palm

 python scripts/convert_jsonlist_to_csv.py --infile json_palm --out data/palm_embeddings.csv --label open_palm
 python scripts/convert_jsonlist_to_csv.py --infile json_palm --out data/palm_embeddings.csv --force-mirror
"""
import json
import argparse
import os
import csv
import sys
import re
import numpy as np

def parse_multi_json_file(path):
    """Return a list of JSON objects parsed from file.
    Handles:
      - a single JSON array: [ {...}, {...} ]
      - a single JSON object
      - newline-delimited JSON (one JSON per line)
      - concatenated JSON objects (tries to find matching braces)
    """
    txt = open(path, "r", encoding="utf-8").read().strip()
    if not txt:
        return []

    # Try loading whole file directly
    try:
        data = json.loads(txt)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except Exception:
        pass

    # Try NDJSON (one JSON per line)
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    objs = []
    ndjson_ok = True
    if len(lines) > 0:
        for ln in lines:
            try:
                objs.append(json.loads(ln))
            except Exception:
                ndjson_ok = False
                break
        if ndjson_ok and objs:
            return objs

    # Fallback: find concatenated JSON objects by scanning braces
    objs = []
    brace_count = 0
    start_idx = None
    for i, ch in enumerate(txt):
        if ch == "{":
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif ch == "}":
            brace_count -= 1
            if brace_count == 0 and start_idx is not None:
                chunk = txt[start_idx:i+1]
                try:
                    objs.append(json.loads(chunk))
                except Exception as e:
                    # skip invalid chunk
                    pass
                start_idx = None
    return objs

def normalize_landmarks_21_xy(lm21, mirror=False):
    """Normalize landmarks: wrist as origin, scale by max euclidean distance, flatten x,y pairs.
    lm21: list of 21 [x,y] coordinates (normalized between 0..1 typically).
    mirror: horizontally flip (x = 1 - x) before normalization.
    Returns list of 42 floats.
    """
    arr = np.array(lm21, dtype=float)  # shape (21,2)
    if arr.shape[0] != 21 or arr.shape[1] != 2:
        raise ValueError("landmark array must be shape (21,2)")
    if mirror:
        arr[:,0] = 1.0 - arr[:,0]
    origin = arr[0].copy()
    rel = arr - origin
    dists = np.linalg.norm(rel, axis=1)
    maxd = float(dists.max())
    if maxd < 1e-6:
        maxd = 1.0
    norm = (rel / maxd).flatten()
    return [float(x) for x in norm]

def process_records(records, out_rows, forced_label=None, force_mirror=False, per_hand="first"):
    """Process list of json records and append normalized rows to out_rows list.
    per_hand: "first" or "all" (which detected hands in record to process).
    """
    for rec in records:
        # record may have structure like the example:
        # rec["landmarks"] -> list of hands, where each hand is list of 21 [x,y]
        landmarks_list = rec.get("landmarks") or rec.get("landmarks2") or rec.get("keypoints") or None
        if landmarks_list is None:
            # try older key names, or maybe the rec itself IS the landmarks list
            if isinstance(rec, list):
                landmarks_list = rec
            else:
                # no landmarks -> skip
                continue

        labels = rec.get("labels") or rec.get("label") or []
        if isinstance(labels, str):
            labels = [labels]
        # choose label for this record
        if forced_label:
            label_to_use = forced_label
        else:
            label_to_use = labels[0] if labels else "unknown"

        # decide mirroring: prefer explicit leading_hand if present
        leading_hand = rec.get("leading_hand", None)
        # For safety: if string "left"/"right" present, decide to mirror left -> True
        mirror_if_left = None
        if leading_hand:
            try:
                if isinstance(leading_hand, str) and leading_hand.lower().startswith("left"):
                    mirror_if_left = True
                else:
                    mirror_if_left = False
            except Exception:
                mirror_if_left = None

        # process per-hand
        for i, hlm in enumerate(landmarks_list):
            # hlm should be 21 pairs (or maybe nested)
            try:
                # sometimes stored as [ [x,y], [x,y], ...]
                if not hlm:
                    continue
                # If each item is a list of two floats already, ok.
                # Otherwise, try to coerce.
                arr = hlm
                if isinstance(arr, dict):
                    # weird case skip
                    continue
                # convert to list of [x,y]
                pts = []
                for p in arr:
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        pts.append([float(p[0]), float(p[1])])
                    else:
                        # unexpected format -> skip hand
                        pts = []
                        break
                if len(pts) != 21:
                    # skip non-21 sets
                    continue
                # determine mirror
                mirror = force_mirror or (mirror_if_left is True)
                vec = normalize_landmarks_21_xy(pts, mirror=mirror)
                out_rows.append([label_to_use] + ["{:.6f}".format(x) for x in vec])
                if per_hand == "first":
                    break
            except Exception as e:
                # skip problematic hand
                continue

def write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    header = ["label"] + [f"x{i}" if j%2==0 else f"y{i}" for i in range(21) for j in (0,1)]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--infile", "-i", required=True, help="input text file containing multiple JSON records")
    p.add_argument("--out", "-o", default="data/palm_embeddings.csv", help="output CSV path")
    p.add_argument("--label", "-l", default=None, help="force label name for all entries (e.g. open_palm)")
    p.add_argument("--force-mirror", action="store_true", help="mirror all landmarks horizontally (useful to canonicalize handedness)")
    p.add_argument("--per-hand", choices=("first","all"), default="first", help="process first detected hand only or all hands per json")
    args = p.parse_args()

    if not os.path.exists(args.infile):
        print("Input file not found:", args.infile)
        sys.exit(1)

    records = parse_multi_json_file(args.infile)
    if not records:
        print("No JSON objects parsed from file.")
        sys.exit(1)

    print(f"Parsed {len(records)} JSON records from {args.infile}")

    out_rows = []
    process_records(records, out_rows, forced_label=args.label, force_mirror=args.force_mirror, per_hand=args.per_hand)

    if not out_rows:
        print("No valid landmark rows produced (check JSON structure).")
        sys.exit(1)

    write_csv(args.out, out_rows)
    print(f"Wrote {len(out_rows)} rows to {args.out}")

if __name__ == "__main__":
    main()
