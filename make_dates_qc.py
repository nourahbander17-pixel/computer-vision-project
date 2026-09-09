"""Generate the synthetic 'dates-qc mini' bundle used by Labs 1-8.

The official SDAIA bundle (12,000 real conveyor frames) was not supplied with the course files,
so this script draws a procedural stand-in that reproduces every mechanic the labs need:
four grades, five boxable defect classes, polygon masks, session-based splits, an evening
session (s09) that is systematically harder, planted label errors, three corrupt annotation
rows, a double-annotated QA sample, and an unlabelled pool for active learning.

Run:  python make_dates_qc.py            (about 1-2 minutes, fixed seed)
Output: data/dates-qc/  (see README_DATA.md written alongside)
"""
import json, hashlib, csv
from pathlib import Path
import numpy as np
import cv2

OUT = Path(__file__).parent / "data" / "dates-qc"
IMG = OUT / "images"
POOL = OUT / "unlabelled_pool"
KEY = OUT / "answer_key"
for d in (IMG, POOL, KEY):
    d.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(212)
S = 256                                   # frame size
GRADES = ["premium", "standard", "substandard", "reject"]      # frozen order
DEFECTS = ["mould", "skin_split", "insect_damage", "sugaring", "foreign_object"]  # frozen order

# session -> (n_frames, brightness, blue tint, blur, split)
SESSIONS = {
    "s01": (100, 1.00, 0, 0, "train"), "s02": (100, 1.05, 0, 0, "train"), "s03": (100, 0.95, 0, 1, "train"),
    "s04": (100, 1.00, 5, 0, "train"), "s05": (100, 1.10, -5, 0, "train"), "s06": (100, 0.90, 0, 0, "train"),
    "s07": (100, 1.00, 0, 0, "train"), "s08": (100, 1.05, 3, 0, "train"),
    "s10": (100, 1.00, 0, 0, "val"), "s11": (100, 0.95, 2, 0, "val"),
    "s09": (100, 0.55, 25, 1, "test"),   # EVENING session: dark, blue, slight blur -> the hidden weak slice
    "s12": (100, 1.00, 0, 0, "test"),
}
POOL_SESSIONS = {"s13": (150, 0.55, 25, 1), "s14": (150, 1.0, 0, 0)}   # unlabelled: half evening frames


def belt_background():
    bg = np.full((S, S, 3), (66, 64, 60), np.uint8)             # RGB
    noise = rng.normal(0, 6, (S, S, 1)).astype(np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    for y in range(0, S, 32):                                     # belt seams
        cv2.line(bg, (0, y + int(rng.integers(0, 8))), (S, y + int(rng.integers(0, 8))), (52, 50, 48), 1)
    return bg


def draw_fruit(img, mask_layer):
    cx, cy = int(rng.integers(45, S - 45)), int(rng.integers(45, S - 45))
    a, b = int(rng.integers(34, 52)), int(rng.integers(22, 34))
    ang = float(rng.uniform(0, 180))
    base = np.array([125, 72, 32]) + rng.integers(-18, 18, 3)
    m = np.zeros((S, S), np.uint8)
    cv2.ellipse(m, (cx, cy), (a, b), ang, 0, 360, 255, -1)
    if (m > 0).sum() == 0:
        return None
    # shading: darker rim, highlight
    fruit = np.zeros_like(img)
    cv2.ellipse(fruit, (cx, cy), (a, b), ang, 0, 360, tuple(int(v) for v in base), -1)
    cv2.ellipse(fruit, (cx, cy), (a, b), ang, 0, 360, tuple(int(v * 0.6) for v in base), 3)
    hx, hy = int(cx - 0.3 * a), int(cy - 0.3 * b)
    cv2.circle(fruit, (hx, hy), max(3, a // 6), tuple(min(255, int(v * 1.35)) for v in base), -1)
    fruit = cv2.GaussianBlur(fruit, (3, 3), 0)
    sel = m > 0
    img[sel] = fruit[sel]
    return {"mask": sel, "center": (cx, cy), "axes": (a, b), "angle": ang}


def draw_defect(img, fruit, kind):
    """Draw one defect instance on a fruit; return its boolean mask."""
    cx, cy = fruit["center"]; a, b = fruit["axes"]
    fm = fruit["mask"]
    m = np.zeros((S, S), np.uint8)
    for _ in range(20):                       # find a point inside the fruit
        px, py = int(cx + rng.uniform(-a * 0.7, a * 0.7)), int(cy + rng.uniform(-b * 0.7, b * 0.7))
        if 0 <= px < S and 0 <= py < S and fm[py, px]:
            break
    if kind == "mould":
        r = int(rng.integers(6, 13))
        cv2.circle(m, (px, py), r, 255, -1)
        cv2.circle(m, (px + int(rng.integers(-4, 4)), py + int(rng.integers(-4, 4))), int(r * 0.7), 255, -1)
        color = (70, 82, 48)
    elif kind == "skin_split":
        L = int(rng.integers(18, 38)); th = float(rng.uniform(0, np.pi))
        x2, y2 = int(px + L * np.cos(th)), int(py + L * np.sin(th))
        cv2.line(m, (px, py), (x2, y2), 255, int(rng.integers(2, 4)))
        color = (205, 175, 125)
    elif kind == "insect_damage":
        for _ in range(int(rng.integers(3, 7))):
            cv2.circle(m, (px + int(rng.integers(-7, 7)), py + int(rng.integers(-7, 7))), int(rng.integers(2, 4)), 255, -1)
        color = (35, 25, 15)
    elif kind == "sugaring":
        r = int(rng.integers(5, 12))
        cv2.ellipse(m, (px, py), (r, int(r * 0.7)), float(rng.uniform(0, 180)), 0, 360, 255, -1)
        color = (232, 226, 200)
    else:  # foreign_object: anywhere in frame, grey-blue plastic fragment
        px, py = int(rng.integers(12, S - 12)), int(rng.integers(12, S - 12))
        w, h = int(rng.integers(6, 11)), int(rng.integers(12, 20))
        pts = np.array([[px, py], [px + w, py + 2], [px + w - 2, py + h], [px - 2, py + h - 3]], np.int32)
        cv2.fillPoly(m, [pts], 255)
        color = (150, 170, 200)
    if kind != "foreign_object":
        m[~fm] = 0                            # clip to the fruit surface
    sel = m > 0
    if sel.sum() < 4:
        return None
    layer = img.copy()
    layer[sel] = color
    alpha = 0.55 if kind == "sugaring" else 0.9
    img[sel] = (alpha * layer[sel] + (1 - alpha) * img[sel]).astype(np.uint8)
    return sel


def apply_session(img, bright, tint, blur):
    out = img.astype(np.float32) * bright
    out[..., 2] += tint; out[..., 0] -= tint * 0.4
    out = np.clip(out, 0, 255).astype(np.uint8)
    if blur:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    out = np.clip(out.astype(np.int16) + rng.normal(0, 3, out.shape).astype(np.int16), 0, 255).astype(np.uint8)
    return out


def mask_to_polygon(mask):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, 0.8, True).reshape(-1, 2)
    return c if len(c) >= 3 else None


def bbox(mask):
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def grade_from(defect_pct, kinds):
    if "mould" in kinds or "foreign_object" in kinds or defect_pct > 8.0:
        return "reject"
    if defect_pct > 3.0 or "insect_damage" in kinds:
        return "substandard"
    if defect_pct > 0.0:
        return "standard"
    return "premium"


def make_frame(bright, tint, blur):
    img = belt_background()
    fruits = []
    for _ in range(int(rng.integers(1, 4))):
        f = draw_fruit(img, None)
        if f is not None:
            fruits.append(f)
    instances = []     # (class, mask)
    kinds = []
    if rng.random() > 0.10:                                     # ~10% defect-free frames
        for f in fruits:
            n_def = rng.choice([0, 1, 2, 3], p=[0.52, 0.33, 0.12, 0.03])
            for _ in range(n_def):
                kind = str(rng.choice(DEFECTS, p=[0.08, 0.34, 0.20, 0.34, 0.04]))
                m = draw_defect(img, f, kind)
                if m is not None:
                    instances.append((kind, m)); kinds.append(kind)
    fruit_px = sum(f["mask"].sum() for f in fruits)
    defect_px = 0
    if fruits:
        union = np.zeros((S, S), bool)
        for k, m in instances:
            if k != "foreign_object":
                union |= m
        defect_px = int(union.sum())
    pct = 100.0 * defect_px / max(fruit_px, 1)
    grade = grade_from(pct, kinds)
    img = apply_session(img, bright, tint, blur)
    return img, fruits, instances, pct, grade


def write_bundle():
    manifest, boxes_raw, seg = [], [], {}
    truth_pct = {}
    idx = 0
    for sid, (n, bright, tint, blur, split) in SESSIONS.items():
        for k in range(n):
            img, fruits, inst, pct, grade = make_frame(bright, tint, blur)
            name = f"{sid}_f{k:04d}.jpg"
            cv2.imwrite(str(IMG / name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
            manifest.append({"file": name, "session_id": sid, "split": split, "grade": grade,
                             "spec_version": "annotation-spec-v1"})
            truth_pct[name] = round(pct, 3)
            polys = []
            for f in fruits:
                p = mask_to_polygon(f["mask"])
                if p is not None:
                    polys.append({"class": "fruit", "polygon": p.tolist()})
            for cls, m in inst:
                x1, y1, x2, y2 = bbox(m)
                boxes_raw.append({"file": name, "class_name": cls, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
                p = mask_to_polygon(m)
                if p is not None:
                    polys.append({"class": cls, "polygon": p.tolist()})
            seg[name] = polys
            idx += 1
    # ---- unlabelled pool (half evening) ----
    pool_truth = {}
    for sid, (n, bright, tint, blur) in POOL_SESSIONS.items():
        for k in range(n):
            img, fruits, inst, pct, grade = make_frame(bright, tint, blur)
            name = f"{sid}_f{k:04d}.jpg"
            cv2.imwrite(str(POOL / name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
            pool_truth[name] = {"session_id": sid, "grade": grade,
                                "boxes": [{"class_name": c, **dict(zip(["x1", "y1", "x2", "y2"], bbox(m)))} for c, m in inst]}

    # ---- plant label noise in v1 grade labels (train split only) ----
    errors = []
    train_rows = [r for r in manifest if r["split"] == "train"]
    # 25 obvious planted errors: premium <-> reject swaps
    obvious = rng.choice(len(train_rows), 25, replace=False)
    for i in obvious:
        r = train_rows[i]; true = r["grade"]
        r["grade"] = "reject" if true == "premium" else "premium"
        errors.append({"file": r["file"], "true_grade": true, "noisy_grade": r["grade"], "kind": "planted_obvious"})
    # ~3% boundary noise: standard <-> substandard
    rest = [i for i in range(len(train_rows)) if i not in set(obvious)]
    for i in rng.choice(rest, int(0.03 * len(train_rows)), replace=False):
        r = train_rows[i]; true = r["grade"]
        if true in ("standard", "substandard"):
            r["grade"] = "substandard" if true == "standard" else "standard"
            errors.append({"file": r["file"], "true_grade": true, "noisy_grade": r["grade"], "kind": "boundary_noise"})
    # ---- plant 3 corrupt rows in the raw box annotations ----
    boxes_raw.append({"file": manifest[3]["file"], "class_name": "mold", "x1": 40, "y1": 40, "x2": 60, "y2": 60})          # unknown class name
    boxes_raw.append({"file": manifest[5]["file"], "class_name": "sugaring", "x1": 120, "y1": 80, "x2": 100, "y2": 95})    # x2 < x1
    boxes_raw.append({"file": manifest[7]["file"], "class_name": "mould", "x1": 200, "y1": 200, "x2": 300, "y2": 290})     # outside the frame
    # ---- double-annotated QA sample (grade labels), disagreement at standard/substandard ----
    qa = []
    for r in rng.choice(train_rows, 200, replace=False):
        a = r["grade"]; b = a
        if a in ("standard", "substandard") and rng.random() < 0.28:
            b = "substandard" if a == "standard" else "standard"
        elif rng.random() < 0.04:
            b = str(rng.choice(GRADES))
        qa.append({"file": r["file"], "annotator_A": a, "annotator_B": b})

    # ---- write ----
    with open(OUT / "manifest_v1.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "session_id", "split", "grade", "spec_version"]); w.writeheader(); w.writerows(manifest)
    with open(OUT / "annotations_boxes_raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "class_name", "x1", "y1", "x2", "y2"]); w.writeheader(); w.writerows(boxes_raw)
    json.dump(seg, open(OUT / "annotations_polygons.json", "w"))
    with open(OUT / "qa_double_annotation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "annotator_A", "annotator_B"]); w.writeheader(); w.writerows(qa)
    with open(KEY / "label_errors_v1.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "true_grade", "noisy_grade", "kind"]); w.writeheader(); w.writerows(errors)
    json.dump(truth_pct, open(KEY / "defect_area_pct_truth.json", "w"))
    json.dump(pool_truth, open(KEY / "unlabelled_pool_truth.json", "w"))
    (OUT / "ANNOTATION_SPEC.md").write_text(SPEC)
    (OUT / "README_DATA.md").write_text(DATA_README)
    print(f"images: {len(manifest)}  boxes: {len(boxes_raw)} (3 corrupt rows planted)  planted grade errors: {len(errors)}  pool: {len(pool_truth)}")
    from collections import Counter
    print("grade distribution (noisy v1):", Counter(r["grade"] for r in manifest))
    print("defect instances:", Counter(b["class_name"] for b in boxes_raw))


SPEC = """# dates-qc annotation spec, version 1 (annotation-spec-v1)

## 1. Grade taxonomy (frozen order)
premium, standard, substandard, reject

Rules: reject if any mould or foreign object is present, or if total defect area exceeds 8 percent of the visible fruit surface (single view).
substandard if defect area is between 3 and 8 percent, or any insect damage is present. standard if any smaller defect is present. premium if no defect.

## 2. Defect classes (frozen order)
mould, skin_split, insect_damage, sugaring, foreign_object. Every class must have a locatable extent (boxable).

## 3. Boxes
Tight axis-aligned box around the visible extent of the defect, in pixel coordinates (x1, y1, x2, y2), origin top-left.

## 4. Polygons and masks
Annotate the outer boundary only; holes are ignored. One polygon per instance. Fruit polygons cover the visible fruit surface.

## 5. Known gaps in v1 (to be fixed in v2)
No boundary rule for where sugaring ends and healthy skin begins. No rule for co-occurring sugaring and mould. No reference gallery.

## 6. QA
10 percent of frames double-annotated. Class kappa below 0.75 stops labelling until the spec is fixed.
"""

DATA_README = """# dates-qc mini bundle (synthetic stand-in for the course dataset)

images/                  1,200 frames, 256x256 RGB JPEG, 12 sessions (s01-s12), 100 each
manifest_v1.csv          file, session_id, split (by session: train s01-s08, val s10-s11, test s09+s12), grade (v1 labels, contain noise), spec_version
annotations_boxes_raw.csv  raw defect boxes in PIXELS from the annotation vendor; contains 3 corrupt rows (Lab 3 finds them)
annotations_polygons.json  per image: fruit and defect polygons in pixels (Lab 4 converts to YOLO-seg)
qa_double_annotation.csv 200 train frames graded by two annotators (Lab 5 agreement analysis)
unlabelled_pool/         300 unlabelled frames (s13 evening, s14 daytime) for Lab 5 active learning
ANNOTATION_SPEC.md       the v1 spec, with known gaps
answer_key/              INSTRUCTOR ONLY: planted label errors, true defect-area percentages, pool ground truth

Session s09 is an evening session (dark, blue-tinted, slight blur) and sits in the test split only.
It is the planted weak slice that Lab 6 must discover.
"""

if __name__ == "__main__":
    import sys, shutil
    write_bundle()
    # Participants run this with --participant: the instructor-only answer key is removed after
    # generation. The instructor keeps it (it is released in class when a lab calls for it).
    if "--participant" in sys.argv:
        shutil.rmtree(KEY, ignore_errors=True)
        print("answer_key/ removed (participant mode)")
