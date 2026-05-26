from pathlib import Path
import ast

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


def test_wholeheart_trainers_use_explicit_nnunet_init_signature():
    source = Path("track4_wholeheart/nnunet_extensions/nnUNetTrainerWholeHeartAug.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    trainer_names = {
        "nnUNetTrainerWholeHeartAug",
        "nnUNetTrainerWholeHeartRHM",
        "nnUNetTrainerWholeHeartRHMMeanTeacher",
    }
    init_methods = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in trainer_names:
            init_methods[node.name] = next(
                child for child in node.body if isinstance(child, ast.FunctionDef) and child.name == "__init__"
            )

    assert set(init_methods) == trainer_names
    for name, init_method in init_methods.items():
        assert init_method.args.vararg is None, f"{name}.__init__ must not use *args"
        assert init_method.args.kwarg is None, f"{name}.__init__ must not use **kwargs"
        assert [arg.arg for arg in init_method.args.args[:5]] == [
            "self",
            "plans",
            "configuration",
            "fold",
            "dataset_json",
        ]


def test_wholeheart_trainers_call_nnunet_init_with_compatible_keywords():
    source = Path("track4_wholeheart/nnunet_extensions/nnUNetTrainerWholeHeartAug.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    trainer_names = {
        "nnUNetTrainerWholeHeartAug",
        "nnUNetTrainerWholeHeartRHM",
        "nnUNetTrainerWholeHeartRHMMeanTeacher",
    }
    expected_keywords = ["plans", "configuration", "fold", "dataset_json", "device"]

    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name not in trainer_names:
            continue
        init_method = next(child for child in node.body if isinstance(child, ast.FunctionDef) and child.name == "__init__")
        super_calls = [
            child
            for child in ast.walk(init_method)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "__init__"
            and isinstance(child.func.value, ast.Call)
            and isinstance(child.func.value.func, ast.Name)
            and child.func.value.func.id == "super"
        ]

        assert len(super_calls) == 1
        super_call = super_calls[0]
        assert super_call.args == [], f"{node.name}.__init__ should call super().__init__ with keywords"
        assert [keyword.arg for keyword in super_call.keywords] == expected_keywords
