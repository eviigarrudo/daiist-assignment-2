"""
The cost model, shared by train.py (which picks the thresholds) and app.py
(which lets you move them). Every figure is defended in REPORT.md.

Unit of decision: one camera frame of a work area. The frame is POSITIVE when
it contains at least one worker without a helmet (a `head` box). The model's
score for a frame is the confidence of its most confident `head` detection.

Three outcomes per frame:
    score <  t_low            -> PASS    nothing happens
    t_low <= score < t_high   -> REVIEW  a safety officer looks at the frame
    score >= t_high           -> FAIL    a supervisor is sent to the worker
"""

from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class Costs:
    # A frame with an unprotected worker that nobody acts on: expected fine
    # share (2% inspection chance x EUR 2,451 minimum serious PPE fine) plus
    # expected injury cost (1 in 1,000 serious head injury x EUR 60,000) plus
    # a small allowance for the norm it sets. Roughly EUR 120.
    miss: float = 120.0
    # Sending a supervisor to a worker: 10 minutes at a loaded EUR 40/h plus
    # the interrupted worker's time. Paid on a real alert and a false one alike.
    dispatch: float = 8.0
    # A safety officer checking one frame on screen: 20 s at EUR 30/h.
    review: float = 0.17
    # The reviewer is good but not perfect on tiny, distant heads.
    reviewer_recall: float = 0.95
    reviewer_false_alarm: float = 0.02
    # Share of production frames that contain an unprotected worker.
    base_rate: float = 0.05


DEFAULT = Costs()

# Review capacity. 8 cameras, one frame every 2 minutes, over a 10-hour shift.
FRAMES_PER_DAY = 2400
REVIEW_SECONDS = 20
# One safety officer can give about 2 hours a day to the review queue.
REVIEWER_HOURS_PER_DAY = 2.0
MAX_REVIEW_SHARE = REVIEWER_HOURS_PER_DAY * 3600 / REVIEW_SECONDS / FRAMES_PER_DAY   # 0.15


def as_dict(costs: Costs = DEFAULT) -> dict:
    return asdict(costs)


def outcome_rates(scores, labels, t_low, t_high):
    """Share of positive and of negative frames that land in each outcome."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    t_high = max(t_high, t_low)
    fail = scores >= t_high
    review = (scores >= t_low) & ~fail
    passed = ~fail & ~review
    pos, neg = labels, ~labels
    rate = lambda mask, group: mask[group].mean() if group.any() else 0.0
    return {
        "pos": {"pass": rate(passed, pos), "review": rate(review, pos), "fail": rate(fail, pos)},
        "neg": {"pass": rate(passed, neg), "review": rate(review, neg), "fail": rate(fail, neg)},
    }


def expected_cost(scores, labels, t_low, t_high, costs: Costs = DEFAULT, base_rate=None):
    """Expected euros per frame, re-weighted to the production base rate.

    Rates are measured separately on positive and negative frames of the
    evaluation split, then mixed at `base_rate`, so the answer does not depend
    on how many positives the dataset happens to contain.
    """
    pi = costs.base_rate if base_rate is None else base_rate
    r = outcome_rates(scores, labels, t_low, t_high)
    c = costs
    # Positive frame: missed if passed; acted on if failed; reviewer catches most.
    pos = (r["pos"]["pass"] * c.miss
           + r["pos"]["fail"] * c.dispatch
           + r["pos"]["review"] * (c.review
                                   + c.reviewer_recall * c.dispatch
                                   + (1 - c.reviewer_recall) * c.miss))
    # Negative frame: free if passed; wasted dispatch if failed.
    neg = (r["neg"]["fail"] * c.dispatch
           + r["neg"]["review"] * (c.review + c.reviewer_false_alarm * c.dispatch))
    review_share = pi * r["pos"]["review"] + (1 - pi) * r["neg"]["review"]
    return {"cost_per_frame": pi * pos + (1 - pi) * neg, "review_share": review_share, "rates": r}


def confusion_counts(scores, labels, t_low, t_high):
    """Raw counts on the evaluation split: rows truth, columns outcome."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    t_high = max(t_high, t_low)
    out = np.where(scores >= t_high, 2, np.where(scores >= t_low, 1, 0))
    table = np.zeros((2, 3), dtype=int)
    for truth in (0, 1):
        for o in (0, 1, 2):
            table[truth, o] = int(((labels == truth) & (out == o)).sum())
    return table  # [[neg pass, neg review, neg fail], [pos pass, pos review, pos fail]]


def reference_policies(costs: Costs = DEFAULT):
    """Per-frame cost of the policies that need no model."""
    c, pi = costs, costs.base_rate
    check_all = c.review + pi * (c.reviewer_recall * c.dispatch + (1 - c.reviewer_recall) * c.miss) \
        + (1 - pi) * c.reviewer_false_alarm * c.dispatch
    return {
        "do_nothing": pi * c.miss,
        "person_checks_everything": check_all,
        "dispatch_to_every_frame": c.dispatch,
    }
