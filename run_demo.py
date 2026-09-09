"""Tamr Vision end-to-end demo. Run:  python run_demo.py
Steps: 1 data  2 export classifier to ONNX (parity)  3 export detector to ONNX  4 run the edge loop  5 write report.html
Everything is read from config/grading_rules.json. Nothing is hard-coded in the decision.
"""
import os, sys, json, time, base64, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE)
import numpy as np, pandas as pd, cv2, torch
from tamr_vision import *
import onnxruntime as ort
from ultralytics import YOLO

N_FRAMES = int(os.environ.get("N_FRAMES", "40"))        # frames per test session in the demo run
ROOT, CK, DIST, RUNS = "data/dates-qc", "checkpoints", "dist", "runs"
os.makedirs(DIST, exist_ok=True); os.makedirs(RUNS, exist_ok=True)
def step(n, msg): print(f"\n==== Step {n}. {msg}", flush=True)

# ------------------------------------------------------------------ 1. data
step(1, "Dataset")
if not os.path.exists(f"{ROOT}/manifest_v1.csv"):
    print("generating data/dates-qc (about 1 to 2 minutes) ..."); subprocess.run([sys.executable, "make_dates_qc.py", "--participant"], check=True)
man = pd.read_csv(f"{ROOT}/manifest_v1.csv"); print("frames:", len(man), "| splits:", man.split.value_counts().to_dict())
rules = json.load(open("config/grading_rules.json")); print("rules:", rules["rules_version"])

# ------------------------------------------------------------------ 2. classifier -> ONNX with parity
step(2, "Export the grade classifier to ONNX and prove parity")
clf, ckpt = load_checkpoint(f"{CK}/grade_resnet18.pt")
va = DatesQCDataset(ROOT, "val", eval_tf())
parity = export_onnx_with_parity(clf, f"{DIST}/grade_resnet18.onnx", va, n=50, eval_mode=True)
print("parity:", parity); assert parity["parity_ok"], "ONNX output differs from PyTorch: do not deploy"
size_fp32 = os.path.getsize(f"{DIST}/grade_resnet18.onnx") / 1e6
size_int8 = quantize_dynamic_int8(f"{DIST}/grade_resnet18.onnx", f"{DIST}/grade_resnet18_int8.onnx")
print(f"classifier ONNX: FP32 {size_fp32:.1f} MB, INT8 {size_int8:.1f} MB")
cls_sess = ort.InferenceSession(f"{DIST}/grade_resnet18_int8.onnx", providers=["CPUExecutionProvider"])

# ------------------------------------------------------------------ 3. detector -> ONNX and a latency benchmark
step(3, "Export the defect detector to ONNX and benchmark it")
import shutil
p0 = YOLO(f"{CK}/defect_y11n.pt").export(format="onnx", imgsz=256, nms=False, opset=17, dynamic=False, simplify=True, verbose=False)
shutil.copy(p0, f"{DIST}/defect_y11n_256.onnx")
det_sess = ort.InferenceSession(f"{DIST}/defect_y11n_256.onnx", providers=["CPUExecutionProvider"]); det_in = det_sess.get_inputs()[0].name
test = man[man.split == "test"].sort_values("file")
frames = [cv2.imread(f"{ROOT}/images/{f}") for f in test.file[:20]]
for _ in range(10): det_sess.run(None, {det_in: det_preprocess(frames[0], 256)})
lat = []
for i in range(60):
    t0 = time.perf_counter(); x = det_preprocess(frames[i % 20], 256); raw = det_sess.run(None, {det_in: x}); decode_and_nms(raw, 0.25, rules["detector_nms_iou"]); lat.append((time.perf_counter() - t0) * 1000)
bench = {"detector_onnx_MB": round(os.path.getsize(f"{DIST}/defect_y11n_256.onnx") / 1e6, 1), "p50_ms": round(float(np.percentile(lat, 50)), 1), "p99_ms": round(float(np.percentile(lat, 99)), 1)}
print("detector ONNX benchmark:", bench)

# ------------------------------------------------------------------ 4. edge loop
step(4, f"Run the edge loop on {N_FRAMES} frames from each test session")
det, seg = YOLO(f"{CK}/defect_y11n.pt"), YOLO(f"{CK}/defect_y11n_seg.pt")
VERSIONS = {"classifier": f"{DIST}/grade_resnet18_int8.onnx", "detector": f"{CK}/defect_y11n.pt", "segmenter": f"{CK}/defect_y11n_seg.pt",
            "dataset": "dates-qc-v1", "rules": rules["rules_version"], "contract": CONTRACT_VERSION}

def classify(bgr):
    logits = cls_sess.run(None, {"image": preprocess_cv2(bgr)})[0][0]; e = np.exp(logits - logits.max()); return e / e.sum()

def grade_frame(det_result, seg_result, cls_probs, rules):
    thr = rules["detector_conf_per_class"]
    dets = [(det_result.names[int(c)], float(s)) for c, s in zip(det_result.boxes.cls, det_result.boxes.conf)]
    dets = [(n, s) for n, s in dets if s >= thr[n]]
    present = {n for n, _ in dets}
    rep = defect_area_report(seg_result); area = max([r["total_defect_pct"] for r in rep], default=0.0)
    if present & set(rules["reject_on_classes"]) or area > rules["area_reject_threshold_pct"]: g, why = "reject", "reject-on-sight class or area over reject threshold"
    elif present & set(rules["substandard_on_classes"]) or area > rules["area_substandard_threshold_pct"]: g, why = "substandard", "substandard class or area over substandard threshold"
    elif present or area > 0: g, why = "standard", "minor defect present"
    else: g, why = GRADES[int(np.argmax(cls_probs))], "classifier fallback (no defect detected)"
    return g, why, dets, area

footage = pd.concat([test[test.session_id == s].head(N_FRAMES) for s in sorted(test.session_id.unique())])
truth = man.set_index("file").grade
logs, thumbs = [], {}
COL = {"mould": (0, 0, 255), "skin_split": (0, 165, 255), "insect_damage": (0, 255, 255), "sugaring": (255, 0, 255), "foreign_object": (255, 0, 0)}
for _, r in footage.iterrows():
    f = r.file; bgr = cv2.imread(f"{ROOT}/images/{f}"); t0 = time.perf_counter()
    probs = classify(bgr); t1 = time.perf_counter()
    d = det.predict(bgr, imgsz=256, conf=0.05, iou=rules["detector_nms_iou"], verbose=False)[0]; t2 = time.perf_counter()
    s = seg.predict(bgr, imgsz=256, conf=0.25, verbose=False)[0]; t3 = time.perf_counter()
    g, why, dets, area = grade_frame(d, s, probs, rules); t4 = time.perf_counter()
    logs.append({"frame_id": f, "session": r.session_id, "decision": g, "truth": truth[f], "reason": why, "eject": g == "reject",
                 "raw_scores": {"classifier": dict(zip(GRADES, probs.round(3).tolist())), "detections": [(n, round(sc, 3)) for n, sc in dets], "max_defect_area_pct": round(area, 2)},
                 "latency_ms": {"classify": round((t1 - t0) * 1000, 1), "detect": round((t2 - t1) * 1000, 1), "segment": round((t3 - t2) * 1000, 1), "decide": round((t4 - t3) * 1000, 2), "total": round((t4 - t0) * 1000, 1)},
                 "versions": VERSIONS})
    vis = bgr.copy()
    for b, c, sc in zip(d.boxes.xyxy.numpy(), d.boxes.cls.numpy(), d.boxes.conf.numpy()):
        n = d.names[int(c)]
        if sc < rules["detector_conf_per_class"][n]: continue
        x1, y1, x2, y2 = map(int, b); cv2.rectangle(vis, (x1, y1), (x2, y2), COL[n], 2); cv2.putText(vis, f"{n} {sc:.2f}", (x1, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COL[n], 1)
    thumbs[f] = base64.b64encode(cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 70])[1]).decode()
json.dump(logs, open(f"{RUNS}/decision_logs.json", "w"), indent=1)
L = pd.DataFrame([{"frame": l["frame_id"], "session": l["session"], "decision": l["decision"], "truth": l["truth"], "total_ms": l["latency_ms"]["total"]} for l in logs])
L["correct"] = L.decision == L.truth
print(L.head(6).to_string(index=False)); print("decisions:", L.decision.value_counts().to_dict())
print(f"accuracy {L.correct.mean():.3f} | p50 {L.total_ms.median():.0f} ms | p99 {L.total_ms.quantile(.99):.0f} ms | budget {rules['latency_budget_ms_end_to_end']} ms")

# ------------------------------------------------------------------ 5. report
step(5, "Write report.html")
per_s = L.groupby("session").agg(n=("frame", "size"), accuracy=("correct", "mean"), p50_ms=("total_ms", "median")).round(3).reset_index()
ct = pd.crosstab(L.truth, L.decision).reindex(index=GRADES, columns=GRADES, fill_value=0)
stage = pd.DataFrame([l["latency_ms"] for l in logs]).describe(percentiles=[.5, .99]).loc[["50%", "99%"]].round(1)
def tbl(df, index=False):
    return df.to_html(index=index, border=0, classes="t")
cards = f"""
<div class=cards>
<div class=card><div class=k>{len(L)}</div><div class=l>frames graded</div></div>
<div class=card><div class=k>{L.correct.mean():.0%}</div><div class=l>system decision accuracy</div></div>
<div class=card><div class=k>{L.total_ms.median():.0f} ms</div><div class=l>pipeline p50 (budget {rules['latency_budget_ms_end_to_end']} ms)</div></div>
<div class=card><div class=k>{int(L.decision.eq('reject').sum())}</div><div class=l>eject signals sent</div></div>
<div class=card><div class=k>{parity['max_diff']:.0e}</div><div class=l>ONNX parity max diff, {parity['flips']} flips</div></div>
</div>"""
gallery = ""
for l in logs:
    ok = "ok" if l["decision"] == l["truth"] else "bad"
    dets = ", ".join(f"{n} {s}" for n, s in l["raw_scores"]["detections"]) or "none"
    gallery += f"""<div class="fr {ok}"><img src="data:image/jpeg;base64,{thumbs[l['frame_id']]}">
    <div class=meta><b>{l['frame_id']}</b><br>decision: <b class={l['decision']}>{l['decision']}</b> &nbsp; truth: {l['truth']}<br>
    {l['reason']}<br>defects: {dets}<br>area: {l['raw_scores']['max_defect_area_pct']} % &nbsp; latency: {l['latency_ms']['total']} ms</div></div>"""
html = f"""<!DOCTYPE html><html lang=en><head><meta charset=utf-8><title>Tamr Vision demo run</title><style>
body{{font-family:Arial,sans-serif;margin:24px;color:#1c2b3a;line-height:1.45}} h1{{color:#1f4e79;margin-bottom:0}} h2{{color:#1f4e79;border-bottom:1.5px solid #1f4e79;padding-bottom:2px;margin-top:28px}}
.sub{{color:#555;margin-bottom:16px}} .cards{{display:flex;gap:12px;flex-wrap:wrap}} .card{{background:#f1f4f7;border-left:4px solid #1f4e79;padding:10px 16px;min-width:150px}}
.k{{font-size:22pt;font-weight:bold}} .l{{font-size:9.5pt;color:#555}} table.t{{border-collapse:collapse;font-size:10.5pt;margin:8px 0}} .t th{{background:#1f4e79;color:#fff;padding:5px 10px;text-align:left}} .t td{{border:1px solid #c8d0d8;padding:5px 10px}}
.flow{{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:10px 0}} .box{{background:#fff8ee;border:1px solid #e08a1e;padding:8px 12px;border-radius:4px;font-size:10pt}} .arrow{{font-size:18pt;color:#1f4e79}}
.gal{{display:flex;flex-wrap:wrap;gap:10px}} .fr{{width:256px;border:2px solid #c8d0d8;padding:4px;font-size:9pt}} .fr.ok{{border-color:#3a8a3a}} .fr.bad{{border-color:#c0392b}} .fr img{{width:256px;display:block}}
.premium{{color:#3a8a3a}} .standard{{color:#1f4e79}} .substandard{{color:#e08a1e}} .reject{{color:#c0392b}} pre{{background:#f1f4f7;padding:10px;font-size:9.5pt;overflow-x:auto}}
</style></head><body>
<h1>Tamr Vision: one run of the complete system</h1><div class=sub>Computer Vision Systems Development (SDA-AIE-212), SDAIA Academy. Generated by run_demo.py on {time.strftime('%Y-%m-%d %H:%M')}.</div>
{cards}
<h2>1. What happens to every frame</h2>
<div class=flow><div class=box>camera frame<br>256 x 256</div><span class=arrow>→</span><div class=box>classifier (ONNX INT8)<br>grade probabilities</div><span class=arrow>→</span><div class=box>detector (YOLO)<br>defect boxes + confidence</div><span class=arrow>→</span><div class=box>segmenter (YOLO seg)<br>defect area percent</div><span class=arrow>→</span><div class=box>decision rules<br>from config/grading_rules.json</div><span class=arrow>→</span><div class=box>grade + eject signal<br>+ JSON log</div></div>
<p>Rules used in this run (the only place thresholds live):</p><pre>{json.dumps(rules, indent=1)}</pre>
<h2>2. Deployment evidence</h2>
<p>Classifier exported to ONNX: max output difference PyTorch vs ONNX {parity['max_diff']:.2e}, decision flips {parity['flips']} of {parity['n']} images. FP32 {size_fp32:.1f} MB, INT8 {size_int8:.1f} MB.</p>
<p>Detector exported to ONNX at 256 px: {bench['detector_onnx_MB']} MB, latency p50 {bench['p50_ms']} ms, p99 {bench['p99_ms']} ms per frame on this CPU.</p>
<h2>3. Latency per stage (ms)</h2>{tbl(stage, index=True)}
<h2>4. Accuracy per session</h2><p>s09 is the evening session (dark, blue, blurred). Compare it with s12.</p>{tbl(per_s)}
<h2>5. Confusion: rows are truth, columns are the system's decision</h2>{tbl(ct, index=True)}
<h2>6. Versions stamped on every log line</h2><pre>{json.dumps(VERSIONS, indent=1)}</pre>
<h2>7. Every frame: green border = correct, red = wrong</h2><div class=gal>{gallery}</div>
<h2>8. One raw log line</h2><pre>{json.dumps(logs[0], indent=1)}</pre>
</body></html>"""
open("report.html", "w").write(html)
print("wrote report.html and runs/decision_logs.json"); print("\nDONE")
