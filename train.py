"""
Assignment 2: training pipeline.

    uv run python main.py train

Start to finish, no manual steps:
  1. download hard-hat-workers v5 from Roboflow Universe
  2. describe it (instances per class, split sizes, object size)
  3. finetune yolo11n from the COCO checkpoint
  4. evaluate it on the test split against two baselines: the COCO model
     zero-shot and a model that predicts nothing
  5. sweep the thresholds on the validation split, pick the cheapest, and
     report what they cost on the test split
  6. save everything app.py needs into outputs/, and the weights into weights/
"""

import json
import os
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from roboflow import Roboflow
from ultralytics import YOLO

import costs as cm
from api_key import load_api_key

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
WEIGHTS = ROOT / "weights" / "hardhat_yolo11n.pt"

WORKSPACE, PROJECT, VERSION = "joseph-nelson", "hard-hat-workers", 5
DATA_DIR = ROOT / "datasets" / f"{PROJECT}-{VERSION}"
CLASSES = ["head", "helmet"]       # head = a worker WITHOUT a helmet: the positive class
POSITIVE = 0

BASE_MODEL = "yolo11n.pt"
EPOCHS = int(os.environ.get("EPOCHS", 25 if torch.cuda.is_available() else 6))   # GPU: 25; an 8 GB laptop: 6
FRACTION = float(os.environ.get("FRACTION", 1.0))   # <1 only for a quick pipeline test
IMGSZ = 416                        # the photos are ~500 px wide; 416 keeps a laptop run to a few hours
BATCH = 16 if torch.cuda.is_available() else 8   # 8 fits an 8 GB Mac; a Colab T4 takes 16
SEED = 0

GRID = np.round(np.arange(0.0, 1.0001, 0.01), 2)


def device() -> str:
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def eval_device() -> str:
    # Evaluating at conf 0.001 on Apple's GPU filled an 8 GB Mac's memory.
    # The CPU is fast enough for inference with a nano model.
    return "cpu" if device() == "mps" else device()


# ---------------------------------------------------------------- data

def download() -> Path:
    if not (DATA_DIR / "data.yaml").exists():
        rf = Roboflow(api_key=load_api_key())
        rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download(
            "yolov11", location=str(DATA_DIR))
    # Roboflow writes relative split paths that Ultralytics resolves against
    # the wrong folder, so write a data.yaml with absolute ones.
    data_yaml = DATA_DIR / "data_abs.yaml"
    data_yaml.write_text(
        f"path: {DATA_DIR}\ntrain: train/images\nval: valid/images\ntest: test/images\n"
        f"nc: {len(CLASSES)}\nnames: {CLASSES}\n")
    return data_yaml


def image_paths(split: str) -> list[Path]:
    return sorted(p for p in (DATA_DIR / split / "images").iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def ground_truth(img: Path) -> tuple[np.ndarray, np.ndarray]:
    """Boxes as normalised xyxy, and their classes."""
    label = img.parent.parent / "labels" / (img.stem + ".txt")
    rows = [line.split() for line in label.read_text().splitlines() if line.strip()] \
        if label.exists() else []
    if not rows:
        return np.zeros((0, 4)), np.zeros(0, dtype=int)
    a = np.array(rows, dtype=float)
    xc, yc, w, h = a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    return np.stack([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2], 1), a[:, 0].astype(int)


def describe() -> dict:
    stats = {}
    for split in ["train", "valid", "test"]:
        imgs = image_paths(split)
        inst = {c: 0 for c in CLASSES}
        imgs_with = {c: 0 for c in CLASSES}
        areas = {c: [] for c in CLASSES}
        for img in imgs:
            boxes, cls = ground_truth(img)
            for k, name in enumerate(CLASSES):
                mask = cls == k
                inst[name] += int(mask.sum())
                imgs_with[name] += int(mask.any())
                b = boxes[mask]
                areas[name] += list((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))
        stats[split] = {
            "images": len(imgs),
            "instances": inst,
            "images_with": imgs_with,
            "positive_image_share": imgs_with["head"] / max(len(imgs), 1),
            "median_box_area_pct": {c: float(np.median(a) * 100) for c, a in areas.items()},
            "share_boxes_under_1pct": {c: float(np.mean(np.array(a) < 0.01)) for c, a in areas.items()},
        }
    return stats


# ---------------------------------------------------------------- evaluation

def predict(model: YOLO, imgs: list[Path], class_map: dict[int, int]) -> list[dict]:
    """Every detection above conf 0.001, mapped to our class ids."""
    preds = []
    for i in range(0, len(imgs), 8):
        for r in model.predict([str(p) for p in imgs[i:i + 8]], conf=0.001, imgsz=IMGSZ,
                               device=eval_device(), verbose=False, max_det=300):
            cls = r.boxes.cls.cpu().numpy().astype(int)
            keep = np.isin(cls, list(class_map))
            preds.append({
                "boxes": r.boxes.xyxyn.cpu().numpy()[keep],
                "conf": r.boxes.conf.cpu().numpy()[keep],
                "cls": np.array([class_map[c] for c in cls[keep]], dtype=int),
            })
    return preds


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def match(preds, gts, k: int, thr: float):
    """Greedy, most confident first: is each class-k detection a true positive?"""
    confs, tps, n_gt = [], [], 0
    for p, (gb, gc) in zip(preds, gts):
        g = gb[gc == k]
        n_gt += len(g)
        m = p["cls"] == k
        boxes, conf = p["boxes"][m], p["conf"][m]
        order = np.argsort(-conf)
        boxes, conf = boxes[order], conf[order]
        taken = np.zeros(len(g), dtype=bool)
        overlaps = iou(boxes, g) if len(g) and len(boxes) else np.zeros((len(boxes), len(g)))
        for j in range(len(boxes)):
            hit = False
            if len(g):
                cand = np.where(~taken & (overlaps[j] >= thr))[0]
                if len(cand):
                    taken[cand[np.argmax(overlaps[j, cand])]] = True
                    hit = True
            tps.append(hit)
        confs += list(conf)
    return np.array(confs), np.array(tps, dtype=bool), n_gt


def average_precision(conf, tp, n_gt) -> float:
    if n_gt == 0 or len(conf) == 0:
        return 0.0
    order = np.argsort(-conf)
    tp = tp[order]
    recall = np.cumsum(tp) / n_gt
    precision = np.cumsum(tp) / np.arange(1, len(tp) + 1)
    r = np.concatenate([[0], recall, [1]])
    p = np.concatenate([[1], precision, [0]])
    p = np.maximum.accumulate(p[::-1])[::-1]
    return float(np.sum((r[1:] - r[:-1]) * p[1:]))


def detection_metrics(preds, gts) -> dict:
    out = {}
    for k, name in enumerate(CLASSES):
        conf, tp, n_gt = match(preds, gts, k, 0.5)
        ap50 = average_precision(conf, tp, n_gt)
        ap5095 = float(np.mean([average_precision(*match(preds, gts, k, t))
                                for t in np.arange(0.5, 0.96, 0.05)]))
        out[name] = {"AP50": ap50, "AP50-95": ap5095, "instances": n_gt}
    out["all"] = {"mAP50": float(np.mean([out[c]["AP50"] for c in CLASSES])),
                  "mAP50-95": float(np.mean([out[c]["AP50-95"] for c in CLASSES]))}
    return out


def detection_table(preds, gts) -> pd.DataFrame:
    """One row per detection with its IoU-0.5 match, so the app can compute
    per-class precision and recall at any threshold."""
    rows = []
    for k, name in enumerate(CLASSES):
        conf, tp, n_gt = match(preds, gts, k, 0.5)
        rows.append(pd.DataFrame({"cls": name, "conf": conf, "tp": tp, "n_gt": n_gt}))
    return pd.concat(rows, ignore_index=True)


def per_class_at(table: pd.DataFrame, t: float) -> dict:
    out = {}
    for name in CLASSES:
        d = table[table.cls == name]
        n_gt = int(d.n_gt.iloc[0]) if len(d) else 0
        kept = d[d.conf >= t]
        tp = int(kept.tp.sum())
        out[name] = {"precision": tp / len(kept) if len(kept) else 0.0,
                     "recall": tp / n_gt if n_gt else 0.0,
                     "TP": tp, "FP": int(len(kept) - tp), "FN": n_gt - tp}
    return out


def frame_scores(preds, gts) -> pd.DataFrame:
    """Frame level: positive if any head; score is the most confident head."""
    return pd.DataFrame({
        "label": [int((gc == POSITIVE).any()) for _, gc in gts],
        "score": [float(p["conf"][p["cls"] == POSITIVE].max()) if (p["cls"] == POSITIVE).any() else 0.0
                  for p in preds],
    })


def roc_auc(labels, scores) -> float:
    labels, scores = np.asarray(labels), np.asarray(scores)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).mean()
    ties = (pos[:, None] == neg[None, :]).mean()
    return float(greater + 0.5 * ties)


# ---------------------------------------------------------------- thresholds

def sweep_single(frames: pd.DataFrame, costs=cm.DEFAULT) -> pd.DataFrame:
    """Model decides alone: one threshold, pass or fail."""
    rows = []
    for t in GRID:
        r = cm.expected_cost(frames.score, frames.label, t, t, costs)
        rows.append({"threshold": t, "cost_per_1000": 1000 * r["cost_per_frame"],
                     "recall": r["rates"]["pos"]["fail"], "false_alarm_rate": r["rates"]["neg"]["fail"]})
    return pd.DataFrame(rows)


def best_band(frames: pd.DataFrame, costs=cm.DEFAULT, max_review_share=1.0) -> dict:
    """Cheapest (t_low, t_high) pair, optionally capped by review capacity."""
    best = None
    for lo in GRID:
        for hi in GRID[GRID >= lo]:
            r = cm.expected_cost(frames.score, frames.label, lo, hi, costs)
            if r["review_share"] > max_review_share:
                continue
            if best is None or r["cost_per_frame"] < best["cost_per_frame"] - 1e-12:
                best = {"t_low": float(lo), "t_high": float(hi),
                        "cost_per_frame": r["cost_per_frame"], "review_share": r["review_share"]}
    return best


def sensitivity(val: pd.DataFrame) -> list[dict]:
    """How the cheapest single threshold moves when the key assumptions move."""
    rows = []
    for miss in [10, 25, 50, 120, 250, 500]:
        c = cm.Costs(miss=miss)
        s = sweep_single(val, c)
        b = s.loc[s.cost_per_1000.idxmin()]
        rows.append({"vary": "miss", "value": miss, "best_threshold": float(b.threshold),
                     "cost_per_1000": float(b.cost_per_1000)})
    for pi in [0.01, 0.02, 0.05, 0.10, 0.20]:
        c = cm.Costs(base_rate=pi)
        s = sweep_single(val, c)
        b = s.loc[s.cost_per_1000.idxmin()]
        rows.append({"vary": "base_rate", "value": pi, "best_threshold": float(b.threshold),
                     "cost_per_1000": float(b.cost_per_1000)})
    return rows


def plot_curve(val_sweep, test_sweep, t_star, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(val_sweep.threshold, val_sweep.cost_per_1000, label="validation (used to choose)")
    ax.plot(test_sweep.threshold, test_sweep.cost_per_1000, "--", label="test (reported)")
    ax.axvline(t_star, color="k", lw=1)
    ax.annotate(f"chosen t = {t_star:.2f}", (t_star, ax.get_ylim()[1] * 0.9), xytext=(6, 0),
                textcoords="offset points")
    ax.set_xlabel("confidence threshold (fail if most confident head >= t)")
    ax.set_ylabel("expected cost, EUR per 1,000 frames")
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- pipeline

def train(data_yaml: Path) -> YOLO:
    run = ROOT / "runs" / "hardhat"
    last = run / "weights" / "last.pt"
    # An interrupted run (laptop closed, app quit) picks up from its last
    # completed epoch instead of starting over. A finished run leaves no
    # results to resume, so a fresh `main.py train` always trains from scratch.
    if os.environ.get("EVAL_ONLY") == "1" and (run / "weights" / "best.pt").exists():
        # Training finished but its evaluation was stopped: evaluate its best.pt.
        shutil.copy(run / "weights" / "best.pt", WEIGHTS)
        return YOLO(str(WEIGHTS))
    if os.environ.get("RESUME") == "1" and last.exists():
        model = YOLO(str(last))
        model.train(resume=True)
    else:
        model = YOLO(BASE_MODEL)
        model.train(data=str(data_yaml), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=device(),
                    seed=SEED, patience=10, fraction=FRACTION, project=str(ROOT / "runs"), name="hardhat",
                    exist_ok=True, plots=True, verbose=False)
    best = Path(model.trainer.best)
    WEIGHTS.parent.mkdir(exist_ok=True)
    shutil.copy(best, WEIGHTS)
    return YOLO(str(WEIGHTS))


def main():
    OUT.mkdir(exist_ok=True)
    data_yaml = download()

    stats = describe()
    (OUT / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))

    model = train(data_yaml)

    # Ultralytics' own test-split evaluation, for its confusion matrix and as a
    # cross-check on the numbers computed below.
    val = model.val(data=str(data_yaml), split="test", imgsz=IMGSZ, device=eval_device(), batch=8,
                    project=str(ROOT / "runs"), name="test_eval", exist_ok=True, plots=True)
    shutil.copy(Path(val.save_dir) / "confusion_matrix.png", OUT / "confusion_matrix_boxes.png")
    official = {"mAP50": float(val.box.map50), "mAP50-95": float(val.box.map),
                "per_class": {CLASSES[int(c)]: {"precision": float(p), "recall": float(r),
                                                "AP50": float(a50), "AP50-95": float(a)}
                              for c, p, r, a50, a in zip(val.box.ap_class_index, val.box.p, val.box.r,
                                                         val.box.ap50, val.box.ap)}}

    coco = YOLO(BASE_MODEL)
    gts = {s: [ground_truth(p) for p in image_paths(s)] for s in ["valid", "test"]}
    preds = {
        "finetuned": {s: predict(model, image_paths(s), {0: 0, 1: 1}) for s in ["valid", "test"]},
        # COCO has no head or helmet class. Its nearest concept is `person`
        # (id 0), so we read every person box as a head, the most generous reading.
        "coco_zero_shot": {s: predict(coco, image_paths(s), {0: 0}) for s in ["valid", "test"]},
        "predict_nothing": {s: [{"boxes": np.zeros((0, 4)), "conf": np.zeros(0), "cls": np.zeros(0, dtype=int)}
                                for _ in gts[s]] for s in ["valid", "test"]},
    }

    frames = {name: {s: frame_scores(preds[name][s], gts[s]) for s in ["valid", "test"]} for name in preds}
    frames["finetuned"]["test"].assign(
        image=[p.name for p in image_paths("test")],
        coco_score=frames["coco_zero_shot"]["test"].score,
    ).to_csv(OUT / "test_frames.csv", index=False)

    # Choose on validation, report on test, so the threshold is not fit to the
    # numbers we report.
    val_sweep = sweep_single(frames["finetuned"]["valid"])
    test_sweep = sweep_single(frames["finetuned"]["test"])
    t_star = float(val_sweep.loc[val_sweep.cost_per_1000.idxmin(), "threshold"])
    unconstrained = best_band(frames["finetuned"]["valid"])
    band = best_band(frames["finetuned"]["valid"], max_review_share=cm.MAX_REVIEW_SHARE)
    test_sweep.assign(val_cost_per_1000=val_sweep.cost_per_1000).to_csv(OUT / "cost_sweep.csv", index=False)
    plot_curve(val_sweep, test_sweep, t_star, OUT / "cost_curve.png")

    table = detection_table(preds["finetuned"]["test"], gts["test"])
    table.to_csv(OUT / "test_detections.csv", index=False)

    results = {}
    for name in preds:
        f = frames[name]["test"]
        single = cm.expected_cost(f.score, f.label, t_star, t_star)
        results[name] = {
            "detection": detection_metrics(preds[name]["test"], gts["test"]),
            "frame_auc": roc_auc(f.label, f.score),
            "frame_accuracy_at_t": float(((f.score >= t_star) == f.label.astype(bool)).mean()),
            "cost_per_1000_at_t": 1000 * single["cost_per_frame"],
            "recall_at_t": single["rates"]["pos"]["fail"],
            "false_alarm_rate_at_t": single["rates"]["neg"]["fail"],
        }
        print(name, json.dumps(results[name], indent=1))

    test_f = frames["finetuned"]["test"]
    band_test = cm.expected_cost(test_f.score, test_f.label, band["t_low"], band["t_high"])
    free_test = cm.expected_cost(test_f.score, test_f.label, unconstrained["t_low"], unconstrained["t_high"])
    summary = {
        "dataset": {"workspace": WORKSPACE, "project": PROJECT, "version": VERSION},
        "training": {"base_model": BASE_MODEL, "epochs": EPOCHS, "imgsz": IMGSZ, "batch": BATCH,
                     "device": device(), "seed": SEED, "best_epoch_weights": str(WEIGHTS.relative_to(ROOT))},
        "positive_class": CLASSES[POSITIVE],
        "costs": cm.as_dict(),
        "chosen_threshold": t_star,
        "chosen_band": band,
        "max_review_share": cm.MAX_REVIEW_SHARE,
        "unconstrained_band": unconstrained,
        "test_at_unconstrained_band": {"cost_per_1000": 1000 * free_test["cost_per_frame"],
                                       "review_share": free_test["review_share"]},
        "test_at_band": {"cost_per_1000": 1000 * band_test["cost_per_frame"],
                         "review_share": band_test["review_share"],
                         "confusion": cm.confusion_counts(test_f.score, test_f.label,
                                                          band["t_low"], band["t_high"]).tolist()},
        "test_confusion_at_t": cm.confusion_counts(test_f.score, test_f.label, t_star, t_star).tolist(),
        "per_class_at_t": per_class_at(table, t_star),
        "ultralytics_test": official,
        "models": results,
        "reference_policies_per_1000": {k: 1000 * v for k, v in cm.reference_policies().items()},
        "sensitivity": sensitivity(frames["finetuned"]["valid"]),
    }
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ["chosen_threshold", "chosen_band", "test_at_band",
                                               "reference_policies_per_1000"]}, indent=2))


if __name__ == "__main__":
    main()
