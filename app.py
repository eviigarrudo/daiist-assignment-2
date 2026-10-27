"""
Assignment 2: Gradio dashboard.

    uv run python main.py app

Reads only what train.py already produced: the committed weights in weights/
and the evaluation outputs in outputs/. Nothing is trained here.
"""

import json
from dataclasses import replace
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ultralytics import YOLO

import costs as cm

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
WEIGHTS = ROOT / "weights" / "hardhat_yolo11n.pt"

model = YOLO(str(WEIGHTS))
metrics = json.loads((OUT / "metrics.json").read_text())
frames = pd.read_csv(OUT / "test_frames.csv")
detections = pd.read_csv(OUT / "test_detections.csv")
sweep = pd.read_csv(OUT / "cost_sweep.csv")

CLASSES = ["head", "helmet"]
T_STAR = metrics["chosen_threshold"]
BAND = metrics["chosen_band"]
OUTCOMES = ["pass", "human review", "fail (dispatch)"]


def costs_from(miss, base_rate):
    return replace(cm.DEFAULT, miss=float(miss), base_rate=float(base_rate))


# ---------------------------------------------------------------- image tab

def predict_image(image, t_low, t_high):
    if image is None:
        return None, "Upload a frame to see the model's decision."
    t_high = max(t_high, t_low)
    result = model.predict(image, conf=max(t_low, 0.01), imgsz=416, verbose=False)[0]
    img = image.copy()
    head_scores = []
    for box, conf, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(),
                              result.boxes.cls.cpu().numpy().astype(int)):
        x1, y1, x2, y2 = box.astype(int)
        if cls == 0:
            head_scores.append(conf)
            colour = (220, 30, 30) if conf >= t_high else (240, 150, 0)
        else:
            colour = (30, 160, 60)
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(img, f"{CLASSES[cls]} {conf:.2f}", (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
    score = max(head_scores, default=0.0)
    if score >= t_high:
        verdict = f"**FAIL: send a supervisor.** Most confident bare head: {score:.2f} (at or above {t_high:.2f})."
    elif score >= t_low:
        verdict = f"**HUMAN REVIEW.** Most confident bare head: {score:.2f}, between {t_low:.2f} and {t_high:.2f}."
    else:
        verdict = f"**PASS.** No bare head scored at or above {t_low:.2f}."
    n_helmets = int((result.boxes.cls == 1).sum())
    verdict += f"\n\nDetected above {t_low:.2f}: {len(head_scores)} bare head(s), {n_helmets} helmet(s). " \
               "Red = bare head at fail level, orange = bare head in the review band, green = helmet."
    return img, verdict


# ---------------------------------------------------------------- operating-point tab

def confusion_figure(t_low, t_high):
    table = cm.confusion_counts(frames.score, frames.label, t_low, t_high)
    fig = go.Figure(go.Heatmap(
        z=table, x=OUTCOMES, y=["no bare head (negative)", "bare head (positive)"],
        text=table, texttemplate="%{text}", colorscale="Blues", showscale=False))
    fig.update_layout(title=f"Test split, {len(frames)} frames: truth vs outcome",
                      height=300, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def curve_figure(t_low, t_high, miss, base_rate):
    c = costs_from(miss, base_rate)
    ts = sweep.threshold.to_numpy()
    cost = [1000 * cm.expected_cost(frames.score, frames.label, t, t, c)["cost_per_frame"] for t in ts]
    best = ts[int(np.argmin(cost))]
    fig = go.Figure()
    fig.add_scatter(x=ts, y=cost, name="test, at current costs", mode="lines")
    if np.isclose(miss, cm.DEFAULT.miss) and np.isclose(base_rate, cm.DEFAULT.base_rate):
        fig.add_scatter(x=ts, y=sweep.val_cost_per_1000, name="validation (used to choose)",
                        mode="lines", line=dict(dash="dot"))
    fig.add_vline(x=T_STAR, line_color="black",
                  annotation_text=f"chosen t = {T_STAR:.2f}", annotation_position="top right")
    fig.add_vrect(x0=t_low, x1=max(t_high, t_low), fillcolor="orange", opacity=0.15, line_width=0,
                  annotation_text="review band", annotation_position="bottom left")
    fig.update_layout(title=f"Cost against threshold, model deciding alone (cheapest at current costs: {best:.2f})",
                      xaxis_title="threshold on the most confident bare head",
                      yaxis_title="EUR per 1,000 frames", height=380,
                      margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.25))
    return fig


def summary_text(t_low, t_high, miss, base_rate, frames_per_day, review_seconds):
    c = costs_from(miss, base_rate)
    r = cm.expected_cost(frames.score, frames.label, t_low, t_high, c)
    alone = cm.expected_cost(frames.score, frames.label, T_STAR, T_STAR, c)
    ref = cm.reference_policies(c)
    reviews = r["review_share"] * frames_per_day
    hours = reviews * review_seconds / 3600
    pos, neg = r["rates"]["pos"], r["rates"]["neg"]
    return f"""
### EUR {1000 * r['cost_per_frame']:,.0f} per 1,000 frames
**{100 * r['review_share']:.1f}%** of frames go to a person: **{reviews:,.0f} a day**, about
**{hours:.1f} reviewer-hours** at {review_seconds:.0f} s each, which is **{hours / 7:.2f} full-time people** (7 productive hours a day).
Capacity assumed: one safety officer, {cm.REVIEWER_HOURS_PER_DAY:.0f} h a day.
{"**Over capacity.**" if hours > cm.REVIEWER_HOURS_PER_DAY + 1e-9 else "Within capacity."}

| | pass | review | fail |
|---|---|---|---|
| bare-head frames | {100 * pos['pass']:.1f}% (missed) | {100 * pos['review']:.1f}% | {100 * pos['fail']:.1f}% |
| compliant frames | {100 * neg['pass']:.1f}% | {100 * neg['review']:.1f}% | {100 * neg['fail']:.1f}% (false alarm) |

| policy | EUR per 1,000 frames |
|---|---|
| **this band ({t_low:.2f} to {max(t_high, t_low):.2f})** | **{1000 * r['cost_per_frame']:,.0f}** |
| model decides alone at t = {T_STAR:.2f} | {1000 * alone['cost_per_frame']:,.0f} |
| do nothing | {1000 * ref['do_nothing']:,.0f} |
| a person checks every frame | {1000 * ref['person_checks_everything']:,.0f} |

Rates are measured on the test split and re-weighted to a base rate of {100 * base_rate:.1f}%.
"""


def per_class_table(t_high):
    rows = []
    det = metrics["models"]["finetuned"]["detection"]
    for name in CLASSES:
        d = detections[detections.cls == name]
        n_gt = int(d.n_gt.iloc[0]) if len(d) else 0
        kept = d[d.conf >= t_high]
        tp = int(kept.tp.sum())
        rows.append({
            "class": name + (" (positive)" if name == "head" else ""),
            "test instances": n_gt,
            "AP50": round(det[name]["AP50"], 3),
            "AP50-95": round(det[name]["AP50-95"], 3),
            f"precision @ {t_high:.2f}": round(tp / len(kept), 3) if len(kept) else 0.0,
            f"recall @ {t_high:.2f}": round(tp / n_gt, 3) if n_gt else 0.0,
            "false boxes": int(len(kept) - tp),
            "missed boxes": n_gt - tp,
        })
    rows.append({"class": "all (mean)", "test instances": int(sum(r["test instances"] for r in rows)),
                 "AP50": round(det["all"]["mAP50"], 3), "AP50-95": round(det["all"]["mAP50-95"], 3)})
    return pd.DataFrame(rows)


def baseline_table():
    rows = []
    for key, label in [("finetuned", "finetuned yolo11n"), ("coco_zero_shot", "COCO yolo11n, zero-shot"),
                       ("predict_nothing", "predict nothing")]:
        m = metrics["models"][key]
        rows.append({"model": label,
                     "mAP50": round(m["detection"]["all"]["mAP50"], 3),
                     "mAP50-95": round(m["detection"]["all"]["mAP50-95"], 3),
                     "head AP50": round(m["detection"]["head"]["AP50"], 3),
                     "frame ROC AUC": round(m["frame_auc"], 3) if m["frame_auc"] == m["frame_auc"] else "n/a",
                     f"EUR / 1,000 @ t={T_STAR:.2f}": round(m["cost_per_1000_at_t"]),
                     "bare-head recall": round(m["recall_at_t"], 3),
                     "false-alarm rate": round(m["false_alarm_rate_at_t"], 3)})
    return pd.DataFrame(rows)


def refresh(t_low, t_high, miss, base_rate, frames_per_day, review_seconds):
    return (confusion_figure(t_low, t_high),
            summary_text(t_low, t_high, miss, base_rate, frames_per_day, review_seconds),
            curve_figure(t_low, t_high, miss, base_rate),
            per_class_table(t_high))


# ---------------------------------------------------------------- layout

with gr.Blocks(title="Hard-hat compliance: operating point") as demo:
    gr.Markdown(
        "# Hard-hat compliance on site cameras\n"
        "Each camera frame is **passed**, sent for **human review**, or **failed** (a supervisor is sent "
        "to the worker). A frame is positive when it shows a worker without a helmet (`head`). "
        "The score is the model's most confident bare-head box. "
        f"Chosen operating point: review from **{BAND['t_low']:.2f}**, fail from **{BAND['t_high']:.2f}**. "
        f"If the model decided alone, the cheapest threshold would be **{T_STAR:.2f}**.")
    with gr.Row():
        t_low = gr.Slider(0, 1, value=BAND["t_low"], step=0.01, label="review from (t_low)")
        t_high = gr.Slider(0, 1, value=BAND["t_high"], step=0.01, label="fail from (t_high)")
    with gr.Accordion("Assumptions (defended in REPORT.md)", open=False):
        with gr.Row():
            miss = gr.Number(value=cm.DEFAULT.miss, label="cost of a missed bare head, EUR")
            base_rate = gr.Slider(0.005, 0.5, value=cm.DEFAULT.base_rate, step=0.005,
                                  label="base rate: share of frames with a bare head")
            frames_per_day = gr.Number(value=cm.FRAMES_PER_DAY, label="frames per day (8 cameras, 1 per 2 min, 10 h)")
            review_seconds = gr.Number(value=cm.REVIEW_SECONDS, label="seconds to review one frame")
    inputs = [t_low, t_high, miss, base_rate, frames_per_day, review_seconds]

    with gr.Tab("Operating point"):
        with gr.Row():
            with gr.Column():
                confusion = gr.Plot()
                summary = gr.Markdown()
            curve = gr.Plot()
    with gr.Tab("Per-class results"):
        per_class = gr.Dataframe(label="Finetuned model on the test split, box level (IoU 0.5)")
        gr.Markdown("Precision and recall follow the **fail** threshold. AP is threshold-free.")
        gr.Dataframe(value=baseline_table(), label="Against the baselines, same test split")
        gr.Image(value=str(OUT / "confusion_matrix_boxes.png"), label="Box-level confusion matrix (Ultralytics)")
    with gr.Tab("Try an image"):
        with gr.Row():
            upload = gr.Image(type="numpy", label="Upload a frame")
            annotated = gr.Image(label="Predictions at the current thresholds")
        verdict = gr.Markdown()
        upload.change(predict_image, [upload, t_low, t_high], [annotated, verdict])
        for s in (t_low, t_high):
            s.release(predict_image, [upload, t_low, t_high], [annotated, verdict])

    outputs = [confusion, summary, curve, per_class]
    for component in inputs:
        component.change(refresh, inputs, outputs)
    demo.load(refresh, inputs, outputs)


if __name__ == "__main__":
    demo.launch()
