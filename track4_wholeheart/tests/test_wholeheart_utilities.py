from pathlib import Path

import numpy as np

from track4_wholeheart.scripts.postprocess_predictions import postprocess_label_array
from track4_wholeheart.scripts.summarize_fold_class_metrics import (
    dice_per_label,
    metrics_from_nnunet_summary,
    parse_training_log_lines,
    safe_mean,
    worst_class,
)


LABEL_NAMES = {
    1: "LV",
    2: "RV",
    3: "LA",
    4: "RA",
    5: "Myo",
    6: "AO",
    7: "PA",
}


def test_postprocess_keeps_largest_component_and_removes_small_island():
    seg = np.zeros((5, 5, 5), dtype=np.uint8)
    seg[0:2, 0:2, 0:2] = 1
    seg[4, 4, 4] = 1
    seg[2:4, 2:4, 2:4] = 2

    cleaned = postprocess_label_array(seg, labels=[1, 2], keep_largest=True)

    assert cleaned[4, 4, 4] == 0
    assert np.count_nonzero(cleaned == 1) == 8
    assert np.count_nonzero(cleaned == 2) == 8


def test_dice_per_label_reports_class_level_failures():
    pred = np.array([1, 1, 2, 2, 0, 0, 3, 0])
    ref = np.array([1, 1, 2, 0, 2, 0, 0, 3])

    scores = dice_per_label(pred, ref, labels=[1, 2, 3])

    assert scores[1] == 1.0
    assert scores[2] == 0.5
    assert scores[3] == 0.0
    assert worst_class(scores, LABEL_NAMES) == ("LA", 0.0)


def test_metrics_from_nnunet_summary_uses_mean_dice_by_label():
    summary = {
        "mean": {
            "1": {"Dice": 0.91},
            "2": {"Dice": 0.82},
            "3": {"Dice": 0.73},
        }
    }

    scores = metrics_from_nnunet_summary(summary, labels=[1, 2, 3])

    assert scores == {1: 0.91, 2: 0.82, 3: 0.73}


def test_parse_training_log_lines_extracts_fold_and_final_pseudo_dice():
    lines = [
        "2026-05-22 12:15:53.394460: Desired fold for training: 4",
        "2026-05-22 12:15:48.044242: Pseudo dice [np.float32(0.95), np.float32(0.92), np.float32(0.91)]",
        "2026-05-22 12:15:48.044513: Yayy! New best EMA pseudo Dice: 0.8992999792098999",
        "2026-05-22 12:20:00.074404: Mean Validation Dice:  0.8941540554478434",
    ]

    parsed = parse_training_log_lines(lines, Path("training_log.txt"))

    assert parsed.fold == 4
    assert parsed.mean_validation_dice == 0.8941540554478434
    assert parsed.final_pseudo_dice == {1: 0.95, 2: 0.92, 3: 0.91}
    assert parsed.best_ema_pseudo_dice == 0.8992999792098999


def test_parse_training_log_lines_accepts_plain_float_pseudo_dice():
    lines = [
        "Desired fold for training: 2",
        "Pseudo dice [0.8, 0.75, 0.6]",
    ]

    parsed = parse_training_log_lines(lines, Path("training_log.txt"))

    assert parsed.final_pseudo_dice == {1: 0.8, 2: 0.75, 3: 0.6}


def test_safe_mean_returns_nan_when_class_absent():
    value = safe_mean([float("nan"), float("nan")])

    assert np.isnan(value)
