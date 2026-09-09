"""tamr_vision.py - single-module version of the course's `tamr_vision/` package.
Everything the eight labs share lives here so participants can read it in one file:
  data      : preprocessing contract (train_tf / eval_tf), DatesQCDataset, stats, throughput
  models    : build_model, freeze_backbone, param_groups, staged training loop, evaluate
  detect    : manifest_to_yolo (with validation), polygons_to_yolo_seg, dataset YAML writers
  seg       : defect_area_report (majority-overlap attribution)
  curate    : audit_labels (loss ranking), cohen_kappa, select_next_batch, release_dataset
  eval      : iou_matrix, match_image, average_precision, slice_report, error_taxonomy
  deploy    : export_onnx_with_parity, quantize_dynamic_int8, benchmark_stages
"""
from __future__ import annotations
import csv, hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
GRADES = ["premium", "standard", "substandard", "reject"]                        # frozen order
DEFECTS = ["mould", "skin_split", "insect_damage", "sugaring", "foreign_object"]  # frozen order
IMG_SIZE = 160                       # classifier input (course uses 224; 160 keeps CPU epochs short)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CONTRACT_VERSION = "transforms@v1(size=160,bilinear,imagenet-stats)"
# =============================================================================
# DATA: the preprocessing contract
# =============================================================================
def eval_tf(size: int = IMG_SIZE):
    """Deterministic path: evaluation harness AND production wrapper import this."""
    return v2.Compose([
        v2.Resize((size, size), interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
        v2.ToDtype(torch.float32, scale=True),           # uint8 [0,255] -> float [0,1]  (before Normalize)
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def train_tf(size: int = IMG_SIZE, strength: float = 1.0):
    """Every transform models a REAL line variation (comment = justification)."""
    return v2.Compose([
        v2.Resize((size, size), interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
        v2.RandomRotation(180),                                        # fruit lands in any pose on the belt
        v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(),            # no canonical orientation
        v2.ColorJitter(brightness=0.30 * strength, contrast=0.20 * strength,
                       saturation=0.20 * strength, hue=0.03 * strength),  # lamp ageing, exposure drift, evening tint
        v2.RandomApply([v2.GaussianBlur(3)], p=0.15),                  # conveyor motion blur
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        v2.RandomErasing(p=0.15, scale=(0.01, 0.05)),                  # partial occlusion by neighbouring fruit
    ])

class DatesQCDataset(Dataset):
    """Split is assigned by SESSION in the manifest, never by frame (the leakage defence)."""
    def __init__(self, root, split, transform=None, manifest="manifest_v1.csv", convert_bgr=True):
        root = Path(root)
        m = pd.read_csv(root / manifest)
        self.rows = m[m["split"] == split].reset_index(drop=True)
        self.root, self.transform, self.convert_bgr = root, transform, convert_bgr
    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        img = cv2.imread(str(self.root / "images" / row["file"]))
        if img is None:                                    # fail loudly, never train on a black frame
            raise FileNotFoundError(row["file"])
        if self.convert_bgr:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)     # convert ONCE at the I/O boundary
        t = torch.from_numpy(img).permute(2, 0, 1)         # HWC uint8 -> CHW uint8
        if self.transform:
            t = self.transform(t)
        return t, GRADES.index(row["grade"])
def dataset_stats(ds, n_max=400):
    """Per-channel mean/std in [0,1] over the first n_max images (raw, no normalisation)."""
    acc = np.zeros(3); acc2 = np.zeros(3); n = 0
    for i in range(min(len(ds), n_max)):
        t, _ = ds[i]
        x = t.float().view(3, -1) / 255.0 if t.dtype == torch.uint8 else t.view(3, -1)
        acc += x.mean(1).numpy(); acc2 += (x ** 2).mean(1).numpy(); n += 1
    mean = acc / n
    return mean.round(3), np.sqrt(acc2 / n - mean ** 2).round(3)
def measure_throughput(ds, batch_size=32, num_workers=0, n_batches=8):
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    it = iter(dl); next(it)                                # warm-up batch
    t0 = time.perf_counter(); n = 0
    for _ in range(n_batches):
        try:
            x, _ = next(it); n += len(x)
        except StopIteration:
            break
    return round(n / (time.perf_counter() - t0), 1)
# =============================================================================
# MODELS: transfer learning + staged fine-tuning
# =============================================================================
def build_model(pretrained=True, arch="resnet18"):
    from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
    if arch == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    else:
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, len(GRADES))    # fresh head, frozen class order
    return model
def freeze_backbone(model):
    """Stage 1: only parameters named fc.* train. Freezing is nothing more than requires_grad flags."""
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("fc.")
def param_groups(model, lr_backbone, lr_head):
    """Stage 2: one optimizer, two learning rates."""
    backbone = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
    head = [p for n, p in model.named_parameters() if n.startswith("fc.")]
    return [{"params": backbone, "lr": lr_backbone}, {"params": head, "lr": lr_head}]
def class_weights(ds):
    counts = ds.rows["grade"].value_counts().reindex(GRADES).fillna(1).values.astype(float)
    w = counts.sum() / (len(GRADES) * counts)
    return torch.tensor(w, dtype=torch.float32)
def train_epoch(model, dl, opt, device, weights=None):
    model.train(); crit = nn.CrossEntropyLoss(weight=weights.to(device) if weights is not None else None)
    total, n = 0.0, 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
        total += loss.item() * len(y); n += len(y)
    return total / n
@torch.no_grad()
def evaluate(model, dl, device):
    """Deterministic eval path: model.eval() inside, per-class recall + macro-F1 (never accuracy alone)."""
    model.eval(); ys, ps, probs = [], [], []
    for x, y in dl:
        logits = model(x.to(device)); pr = logits.softmax(1).cpu()
        ps.append(pr.argmax(1)); ys.append(y); probs.append(pr)
    y = torch.cat(ys).numpy(); p = torch.cat(ps).numpy(); pr = torch.cat(probs).numpy()
    from sklearn.metrics import f1_score, recall_score, confusion_matrix
    return {"macro_f1": round(float(f1_score(y, p, average="macro")), 4),
            "accuracy": round(float((y == p).mean()), 4),
            "per_class_recall": dict(zip(GRADES, recall_score(y, p, average=None, labels=range(len(GRADES))).round(3).tolist())),
            "confusion": confusion_matrix(y, p, labels=range(len(GRADES))).tolist(),
            "y": y, "pred": p, "probs": pr}
def staged_finetune(model, train_dl, val_dl, device, weights=None, stage1_epochs=2, stage2_epochs=4,
                    lr_head=1e-3, lr_backbone=1e-4, patience=2, uniform_lr=None, ckpt_path=None, log=print):
    """Head-only warm-up, then discriminative learning rates with early stopping on val macro-F1.
    uniform_lr: if set, stage 2 uses that single LR for everything (the catastrophic-forgetting demo)."""
    history = []
    freeze_backbone(model)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr_head)
    for e in range(stage1_epochs):
        loss = train_epoch(model, train_dl, opt, device, weights); m = evaluate(model, val_dl, device)
        history.append({"stage": 1, "epoch": e + 1, "loss": round(loss, 4), "macro_f1": m["macro_f1"], "reject_recall": m["per_class_recall"]["reject"]})
        log(f"stage1 epoch {e+1}: loss={loss:.3f} val macro_f1={m['macro_f1']:.3f} reject_recall={m['per_class_recall']['reject']:.2f}")
    for p in model.parameters():
        p.requires_grad = True
    if uniform_lr is not None:
        opt = torch.optim.AdamW(model.parameters(), lr=uniform_lr)
    else:
        opt = torch.optim.AdamW(param_groups(model, lr_backbone, lr_head))
    best, wait, best_state = -1.0, 0, None
    for e in range(stage2_epochs):
        loss = train_epoch(model, train_dl, opt, device, weights); m = evaluate(model, val_dl, device)
        history.append({"stage": 2, "epoch": e + 1, "loss": round(loss, 4), "macro_f1": m["macro_f1"], "reject_recall": m["per_class_recall"]["reject"]})
        log(f"stage2 epoch {e+1}: loss={loss:.3f} val macro_f1={m['macro_f1']:.3f} reject_recall={m['per_class_recall']['reject']:.2f}")
        if m["macro_f1"] > best:
            best, wait = m["macro_f1"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if ckpt_path:
                torch.save({"state_dict": best_state, "classes": GRADES, "contract": CONTRACT_VERSION,
                            "val_metrics": {k: m[k] for k in ("macro_f1", "accuracy", "per_class_recall")}}, ckpt_path)
        elif (wait := wait + 1) >= patience:
            log("early stop"); break
    if best_state:
        model.load_state_dict(best_state)
    return pd.DataFrame(history), best
def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    assert ckpt["classes"] == GRADES, "class order in checkpoint differs from GRADES: refuse to serve"
    sd = {k: (v.float() if v.is_floating_point() else v) for k, v in ckpt["state_dict"].items()}   # fp16-stored weights -> fp32
    model = build_model(pretrained=False); model.load_state_dict(sd); model.eval()
    return model, ckpt
# =============================================================================
# DETECT / SEG: converters with validation
# =============================================================================
def manifest_to_yolo(root, out_dir, boxes_csv="annotations_boxes_raw.csv", manifest="manifest_v1.csv",
                     img_size=256, strict=True, splits=("train", "val", "test")):
    """Pixel boxes -> YOLO txt (class cx cy w h, normalised). Validates every row; returns the bad rows.
    With strict=True the bad rows are reported and SKIPPED (the data must be fixed upstream, not the assertions)."""
    root, out = Path(root), Path(out_dir)
    m = pd.read_csv(root / manifest).set_index("file")
    boxes = pd.read_csv(root / boxes_csv)
    bad = []
    per_file = {f: [] for f in m.index}
    for _, r in boxes.iterrows():
        problems = []
        if r["class_name"] not in DEFECTS: problems.append(f"unknown class '{r['class_name']}'")
        if not (r["x1"] < r["x2"] and r["y1"] < r["y2"]): problems.append("inverted box")
        if r[["x1", "y1", "x2", "y2"]].max() > img_size or r[["x1", "y1", "x2", "y2"]].min() < 0: problems.append("outside frame")
        if problems:
            bad.append({**r.to_dict(), "problem": "; ".join(problems)})
            continue
        cx, cy = (r.x1 + r.x2) / 2 / img_size, (r.y1 + r.y2) / 2 / img_size
        w, h = (r.x2 - r.x1) / img_size, (r.y2 - r.y1) / img_size
        assert 0 <= cx <= 1 and 0 <= w <= 1
        per_file[r["file"]].append(f"{DEFECTS.index(r['class_name'])} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    for split in splits:
        (out / "images" / split).mkdir(parents=True, exist_ok=True); (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    import shutil
    for f, lines in per_file.items():
        split = m.loc[f, "split"]
        if split not in splits: continue
        dst = out / "images" / split / f
        if not dst.exists():
            try: dst.symlink_to((root / "images" / f).resolve())
            except OSError: shutil.copy(root / "images" / f, dst)
        (out / "labels" / split / f.replace(".jpg", ".txt")).write_text("\n".join(lines))
    yaml = f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(DEFECTS))
    (out / "dates-det.yaml").write_text(yaml)
    return pd.DataFrame(bad)
def polygons_to_yolo_seg(root, out_dir, polys_json="annotations_polygons.json", manifest="manifest_v1.csv", img_size=256):
    """Polygons (pixels) -> YOLO-seg txt (class x1 y1 x2 y2 ... normalised). Classes: fruit + 5 defects."""
    root, out = Path(root), Path(out_dir)
    names = ["fruit"] + DEFECTS
    m = pd.read_csv(root / manifest).set_index("file")
    polys = json.load(open(root / polys_json))
    import shutil
    degenerate = 0
    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True); (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    for f, items in polys.items():
        split = m.loc[f, "split"]; lines = []
        for it in items:
            pts = np.array(it["polygon"], float)
            if len(pts) < 3: degenerate += 1; continue
            flat = " ".join(f"{v:.5f}" for v in (pts / img_size).clip(0, 1).ravel())
            lines.append(f"{names.index(it['class'])} {flat}")
        dst = out / "images" / split / f
        if not dst.exists():
            try: dst.symlink_to((root / "images" / f).resolve())
            except OSError: shutil.copy(root / "images" / f, dst)
        (out / "labels" / split / f.replace(".jpg", ".txt")).write_text("\n".join(lines))
    (out / "dates-seg.yaml").write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))
    return degenerate
def defect_area_report(result, defect_classes=("mould", "skin_split", "insect_damage", "sugaring")):
    """Per-fruit defect-area percentage from a YOLO-seg result, majority-overlap attribution."""
    if result.masks is None:
        return []
    names = result.names
    masks = result.masks.data.cpu().numpy().astype(bool)
    H, W = result.orig_shape
    if masks.shape[1:] != (H, W):
        masks = np.stack([cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool) for m in masks])
    classes = [names[int(c)] for c in result.boxes.cls]
    fruit = [i for i, c in enumerate(classes) if c == "fruit"]
    defects = [i for i, c in enumerate(classes) if c in defect_classes]
    report = []
    for fi in fruit:
        fruit_px = masks[fi].sum()
        if fruit_px == 0: continue
        areas = {c: 0.0 for c in defect_classes}
        for di in defects:
            overlap = (masks[di] & masks[fi]).sum()
            if overlap / max(masks[di].sum(), 1) >= 0.5:          # majority-overlap rule
                areas[classes[di]] += overlap / fruit_px
        report.append({"fruit_id": fi, "fruit_px": int(fruit_px), **{k: round(100 * v, 2) for k, v in areas.items()},
                       "total_defect_pct": round(100 * sum(areas.values()), 2)})
    return report            # the 8% rule is applied downstream from config, never here

# =============================================================================
# CURATE
# =============================================================================
@torch.no_grad()
def audit_labels(model, ds, device="cpu", batch_size=64):
    """Rank training items by per-item loss under the current model; label errors concentrate at the top."""
    model.eval(); crit = nn.CrossEntropyLoss(reduction="none"); rec = []
    dl = DataLoader(ds, batch_size=batch_size)
    i = 0
    for x, y in dl:
        logits = model(x.to(device)); loss = crit(logits, y.to(device)).cpu().numpy()
        conf, pred = logits.softmax(1).max(1)
        for k in range(len(y)):
            r = ds.rows.iloc[i + k]
            rec.append({"file": r["file"], "session_id": r["session_id"], "label": GRADES[int(y[k])],
                        "pred": GRADES[int(pred[k])], "conf": round(float(conf[k]), 3), "loss": round(float(loss[k]), 4)})
        i += len(y)
    return pd.DataFrame(rec).sort_values("loss", ascending=False).reset_index(drop=True)
def cohen_kappa(a, b, labels=GRADES):
    a, b = pd.Categorical(a, categories=labels), pd.Categorical(b, categories=labels)
    cm = pd.crosstab(a, b, dropna=False).reindex(index=labels, columns=labels, fill_value=0).values.astype(float)
    n = cm.sum(); po = np.trace(cm) / n
    pe = (cm.sum(0) * cm.sum(1)).sum() / n ** 2
    return float((po - pe) / (1 - pe)), pd.DataFrame(cm.astype(int), index=labels, columns=labels)
@torch.no_grad()
def score_pool(model, pool_dir, device="cpu", size=IMG_SIZE):
    """Uncertainty (entropy) of the classifier on unlabelled frames."""
    model.eval(); tf = eval_tf(size); rows = []
    for p in sorted(Path(pool_dir).glob("*.jpg")):
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        x = tf(torch.from_numpy(img).permute(2, 0, 1)).unsqueeze(0)
        pr = model(x.to(device)).softmax(1)[0].cpu().numpy()
        ent = float(-(pr * np.log(pr + 1e-9)).sum())
        rows.append({"file": p.name, "session_id": p.name.split("_")[0], "entropy": round(ent, 4), "pred": GRADES[int(pr.argmax())],
                     "mean_intensity": round(float(img.mean()), 1)})
    return pd.DataFrame(rows).sort_values("entropy", ascending=False).reset_index(drop=True)
def select_next_batch(scores, budget, per_session_cap=None):
    """Uncertainty sampling with a diversity cap per session (stops collapse onto one failure mode)."""
    chosen, counts = [], {}
    for _, r in scores.iterrows():
        if per_session_cap and counts.get(r["session_id"], 0) >= per_session_cap: continue
        chosen.append(r["file"]); counts[r["session_id"]] = counts.get(r["session_id"], 0) + 1
        if len(chosen) >= budget: break
    return chosen
def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def release_dataset(root, version, manifest_name, spec_version, changelog):
    """Immutable release: manifest with per-file hashes + release hash + changelog. v3 supersedes, never edits."""
    root = Path(root); rel = root / "releases" / version; rel.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(root / manifest_name)
    m["sha256"] = [sha256(root / "images" / f) if (root / "images" / f).exists() else sha256(root / "unlabelled_pool" / f) for f in m["file"]]
    m.to_csv(rel / "manifest.csv", index=False, lineterminator="\n")
    info = {"version": version, "spec_version": spec_version, "release_hash": sha256(rel / "manifest.csv"), "n_images": int(len(m)),
            "changelog": changelog}
    json.dump(info, open(rel / "RELEASE.json", "w"), indent=2)
    return info
def verify_split_freeze(root, v_old, v_new):
    a = pd.read_csv(Path(root) / "releases" / v_old / "manifest.csv"); b = pd.read_csv(Path(root) / "releases" / v_new / "manifest.csv")
    old_test = a[a.split == "test"].set_index("file"); new = b.set_index("file")
    missing = [f for f in old_test.index if f not in new.index]
    moved = [f for f in old_test.index if f in new.index and new.loc[f, "split"] != "test"]
    return {"ok": not missing and not moved, "missing_from_new": missing, "moved_out_of_test": moved}
# =============================================================================
# EVAL: matching and AP from first principles
# =============================================================================
def iou_matrix(a, b):
    """a: (P,4) b: (G,4) in x1y1x2y2 -> (P,G) IoU."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)
def match_image(preds, gts, iou_thr=0.5):
    """Greedy matching in confidence order, same class only, each GT consumed once.
    preds: dict(boxes (P,4), cls (P,), conf (P,)); gts: dict(boxes (G,4), cls (G,)).
    Returns tp flags in input order and the indices of unmatched GT (misses)."""
    P = len(preds["conf"]); G = len(gts["cls"])
    order = np.argsort(-np.asarray(preds["conf"]))
    tp = np.zeros(P, bool); consumed = np.zeros(G, bool)
    iou = iou_matrix(preds["boxes"], gts["boxes"])
    matched_gt = -np.ones(P, int)
    for p in order:
        cand = np.where((np.asarray(gts["cls"]) == preds["cls"][p]) & ~consumed)[0]
        if len(cand) == 0:
            continue                                   # false positive
        best = cand[np.argmax(iou[p, cand])]
        if iou[p, best] >= iou_thr:
            tp[p], consumed[best], matched_gt[p] = True, True, best
    return tp, np.where(~consumed)[0], matched_gt, iou
def average_precision(tp, conf, n_gt):
    """101-point interpolated AP (COCO convention)."""
    if n_gt == 0:
        return float("nan")
    order = np.argsort(-np.asarray(conf)); tp = np.asarray(tp)[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(~tp)
    recall = ctp / n_gt; prec = ctp / np.clip(ctp + cfp, 1, None)
    return float(np.mean([prec[recall >= r].max() if (recall >= r).any() else 0.0 for r in np.linspace(0, 1, 101)]))


def evaluate_detections(pred_df, gt_df, classes=DEFECTS, iou_thr=0.5, conf_thr=0.0):
    """pred_df: file, cls, conf, x1,y1,x2,y2 ; gt_df: file, cls, x1,y1,x2,y2 (cls as int index).
    Returns per-class AP, recall at conf_thr, plus a per-prediction table with match flags."""
    if len(pred_df) == 0:
        pred_df = pd.DataFrame(columns=["file", "cls", "conf", "x1", "y1", "x2", "y2"])
    pred_df = pred_df[pred_df["conf"] >= conf_thr].copy()
    files = sorted(set(gt_df["file"]) | set(pred_df["file"]))
    rows, misses = [], []
    for f in files:
        p = pred_df[pred_df["file"] == f]; g = gt_df[gt_df["file"] == f]
        preds = {"boxes": p[["x1", "y1", "x2", "y2"]].values.astype(float), "cls": p["cls"].values.astype(int), "conf": p["conf"].values.astype(float)}
        gts = {"boxes": g[["x1", "y1", "x2", "y2"]].values.astype(float), "cls": g["cls"].values.astype(int)}
        tp, unmatched, mg, iou = match_image(preds, gts, iou_thr)
        for k in range(len(p)):
            best_iou = float(iou[k].max()) if iou.shape[1] else 0.0
            rows.append({"file": f, "cls": int(p["cls"].values[k]), "conf": float(p["conf"].values[k]), "tp": bool(tp[k]), "best_iou_any": best_iou,
                         "same_class_gt_exists": bool((g["cls"].values == p["cls"].values[k]).any())})
        for u in unmatched:
            misses.append({"file": f, "cls": int(g["cls"].values[u])})
    P = pd.DataFrame(rows, columns=["file", "cls", "conf", "tp", "best_iou_any", "same_class_gt_exists"]); M = pd.DataFrame(misses, columns=["file", "cls"])
    P["tp"] = P["tp"].astype(bool)
    out = []
    for ci, cname in enumerate(classes):
        n_gt = int((gt_df["cls"] == ci).sum()); pc = P[P["cls"] == ci]
        ap = average_precision(pc["tp"].values, pc["conf"].values, n_gt) if len(pc) else 0.0
        rec = float(pc["tp"].sum() / n_gt) if n_gt else float("nan")
        out.append({"class": cname, "n_gt": n_gt, "AP50": round(ap, 4), "recall_at_conf": round(rec, 4), "n_pred": len(pc), "fp": int((~pc["tp"]).sum()) if len(pc) else 0})
    return pd.DataFrame(out), P, M
def error_taxonomy(P, M, gt_df, iou_thr=0.5):
    """Five buckets: miss, misclassification, localisation, duplicate, background. Returns error mass shares."""
    fp = P[~P["tp"]]
    counts = {"miss": len(M), "misclassification": 0, "localisation": 0, "duplicate": 0, "background": 0}
    for _, r in fp.iterrows():
        if r.best_iou_any >= iou_thr and not r.same_class_gt_exists: counts["misclassification"] += 1
        elif r.best_iou_any >= iou_thr: counts["duplicate"] += 1
        elif r.best_iou_any >= 0.1: counts["localisation"] += 1
        else: counts["background"] += 1
    total = max(sum(counts.values()), 1)
    return {k: round(v / total, 3) for k, v in counts.items()}, counts
def slice_report(pred_df, gt_df, key, iou_thr=0.5, conf_thr=0.25, min_n=50):
    """Recall per slice (key: a function file -> slice name). Slices under min_n GT are flagged under-powered."""
    slices = sorted({key(f) for f in gt_df["file"]})
    rows = []
    for s in slices:
        gf = gt_df[np.array([key(f) == s for f in gt_df["file"]], bool)]; pf = pred_df[np.array([key(f) == s for f in pred_df["file"]], bool)] if len(pred_df) else pred_df
        tab, P, M = evaluate_detections(pf, gf, iou_thr=iou_thr, conf_thr=conf_thr)
        n = int(tab["n_gt"].sum()); rec = 1 - len(M) / max(n, 1)
        rows.append({"slice": s, "n_gt": n, "recall": round(rec, 3), "mean_AP50": round(float(tab["AP50"].mean()), 3), "powered": n >= min_n})
    return pd.DataFrame(rows)
# =============================================================================
# DEPLOY
# =============================================================================
def export_onnx_with_parity(model, ckpt_path_out, ds, n=50, size=IMG_SIZE, eval_mode=True, opset=17):
    """Export in eval mode and verify parity on REAL images: max diff < 1e-4 AND zero decision flips."""
    import onnxruntime as ort
    if eval_mode:
        model.eval()
    else:
        model.train()          # the planted bug: exporting in train mode
    dummy = torch.randn(1, 3, size, size)
    torch.onnx.export(model, dummy, ckpt_path_out, opset_version=opset, input_names=["image"], output_names=["logits"], dynamo=False)
    model.eval()
    sess = ort.InferenceSession(ckpt_path_out, providers=["CPUExecutionProvider"])
    max_diff, flips = 0.0, 0
    for i in range(min(n, len(ds))):
        x, _ = ds[i]; x = x.unsqueeze(0)
        with torch.no_grad():
            ref = model(x).numpy()
        out = sess.run(None, {"image": x.numpy()})[0]
        max_diff = max(max_diff, float(np.abs(ref - out).max())); flips += int(ref.argmax() != out.argmax())
    return {"max_diff": max_diff, "flips": flips, "n": min(n, len(ds)), "parity_ok": max_diff < 1e-4 and flips == 0}
def quantize_dynamic_int8(onnx_in, onnx_out):
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(onnx_in, onnx_out, weight_type=QuantType.QInt8)
    return Path(onnx_out).stat().st_size / 1e6
def preprocess_cv2(img_bgr, size=IMG_SIZE):
    """Production reimplementation of eval_tf() in OpenCV. Must be equivalence-tested against eval_tf()."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)   # antialiased downscale ~ torchvision antialias bilinear
    x = rgb.astype(np.float32) / 255.0
    x = (x - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
    return x.transpose(2, 0, 1)[None]
def benchmark_stages(sess, frames_bgr, input_name="image", n_runs=300, warmup=20, size=IMG_SIZE, postprocess=None):
    """Honest benchmark: warm-up discarded, p50/p95/p99 per stage over n_runs on real frames."""
    for _ in range(warmup):
        sess.run(None, {input_name: preprocess_cv2(frames_bgr[0], size)})
    stages = {"pre": [], "infer": [], "post": []}
    for i in range(n_runs):
        f = frames_bgr[i % len(frames_bgr)]
        t0 = time.perf_counter(); x = preprocess_cv2(f, size); t1 = time.perf_counter()
        raw = sess.run(None, {input_name: x}); t2 = time.perf_counter()
        _ = postprocess(raw) if postprocess else int(np.argmax(raw[0])); t3 = time.perf_counter()
        stages["pre"].append(t1 - t0); stages["infer"].append(t2 - t1); stages["post"].append(t3 - t2)
    rows = []
    for k, v in stages.items():
        ms = np.array(v) * 1000
        rows.append({"stage": k, "p50_ms": round(np.percentile(ms, 50), 2), "p95_ms": round(np.percentile(ms, 95), 2), "p99_ms": round(np.percentile(ms, 99), 2)})
    total = (np.array(stages["pre"]) + np.array(stages["infer"]) + np.array(stages["post"])) * 1000
    rows.append({"stage": "TOTAL", "p50_ms": round(np.percentile(total, 50), 2), "p95_ms": round(np.percentile(total, 95), 2), "p99_ms": round(np.percentile(total, 99), 2)})
    return pd.DataFrame(rows)
# =============================================================================
# DEPLOY (detector helpers): decode + NMS in code, static INT8 with calibration, ONNX predictions for the harness
# =============================================================================
def det_preprocess(bgr, size):
    x = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (size, size)).astype(np.float32) / 255.0
    return x.transpose(2, 0, 1)[None]
def decode_and_nms(raw, conf_thr=0.25, iou_thr=0.45):
    """Ultralytics ONNX output (1, 4+nc, N) -> list of (cls, conf, xyxy). NMS in CODE so the operating point stays in config."""
    out = raw[0][0]
    boxes, scores = out[:4].T, out[4:].T
    cls = scores.argmax(1); sc = scores.max(1); keep = sc > conf_thr
    if keep.sum() == 0:
        return []
    b = boxes[keep]; xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
    idx = cv2.dnn.NMSBoxes(np.stack([xyxy[:, 0], xyxy[:, 1], xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]], 1).tolist(), sc[keep].tolist(), conf_thr, iou_thr)
    return [(int(cls[keep][i]), float(sc[keep][i]), xyxy[i]) for i in np.array(idx).ravel()]
def quantize_static_int8(onnx_in, onnx_out, calib_frames_bgr, size, input_name="images", keep_float_substring="model.23"):
    """Static INT8 (QDQ, per-channel) with a calibration set. The calibration frames must be REPRESENTATIVE
    (stratified by session / lighting), or the quantised model collapses on the slices it never saw.
    keep_float_substring: nodes whose name contains it stay in float. For YOLO11 that is the detection head
    (model.23: DFL and box decoding); quantising it destroys the boxes (mAP -> 0) while the backbone and neck quantise cleanly."""
    from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat, CalibrationMethod
    from onnxruntime.quantization.shape_inference import quant_pre_process
    pre = str(onnx_out).replace(".onnx", "_pre.onnx"); quant_pre_process(str(onnx_in), pre)
    class Reader(CalibrationDataReader):
        def __init__(self): self.it = iter(calib_frames_bgr)
        def get_next(self):
            f = next(self.it, None)
            return None if f is None else {input_name: det_preprocess(f, size)}
    import onnx
    exclude = [n.name for n in onnx.load(pre).graph.node if keep_float_substring and keep_float_substring in n.name]
    quantize_static(pre, str(onnx_out), Reader(), quant_format=QuantFormat.QDQ, per_channel=True, nodes_to_exclude=exclude,
                    weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8, calibrate_method=CalibrationMethod.MinMax)
    Path(pre).unlink(missing_ok=True)
    return Path(onnx_out).stat().st_size / 1e6
def onnx_detect_predictions(sess, files, root, size, conf_thr=0.02, iou_thr=0.45, frame_size=256):
    """Run an exported detector over files and return the prediction table the Lab 6 harness consumes (pixel coords of the original frame)."""
    inp = sess.get_inputs()[0].name; rows = []
    for f in files:
        bgr = cv2.imread(str(Path(root) / "images" / f)); raw = sess.run(None, {inp: det_preprocess(bgr, size)})
        scale = frame_size / size
        for c, s, b in decode_and_nms(raw, conf_thr, iou_thr):
            rows.append({"file": f, "cls": c, "conf": s, "x1": b[0] * scale, "y1": b[1] * scale, "x2": b[2] * scale, "y2": b[3] * scale})
    return pd.DataFrame(rows, columns=["file", "cls", "conf", "x1", "y1", "x2", "y2"])
