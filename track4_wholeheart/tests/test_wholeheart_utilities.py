from pathlib import Path
import ast

import numpy as np
import torch

from track4_wholeheart.nnunet_extensions.unlabeled_pool import WholeHeartRawUnlabeledPool
from track4_wholeheart.nnunet_extensions.ct_window_augmentation import (
    apply_ct_window_rescale_torch,
    ct_hu_window_to_normalized,
)
from track4_wholeheart.scripts.postprocess_predictions import (
    build_class_aware_hd_rules,
    postprocess_label_array,
)
from track4_wholeheart.scripts.refine_mr_with_medsam2 import (
    central_bbox_prompt,
    constrained_candidate,
    normalize_volume_to_uint8,
    should_accept_candidate,
)
from track4_wholeheart.scripts.summarize_fold_class_metrics import (
    dice_per_label,
    metrics_from_nnunet_summary,
    parse_training_log_lines,
    safe_mean,
    worst_class,
)
from track4_wholeheart.scripts.evaluate_segmentation_metrics import compute_binary_metrics, crop_to_union_foreground
from track4_wholeheart.scripts.check_prediction_sanity import (
    GeometrySignature,
    compare_geometry,
    check_array_labels,
)
from track4_wholeheart.scripts.postprocess_ct_aopa_soft import (
    adaptive_seed_threshold,
    component_score_cleanup,
    vessel_hysteresis_growing,
)
from track4_wholeheart.scripts.compare_metric_summaries import compare_summary_rows
from track4_wholeheart.scripts.ensemble_probabilities import (
    argmax_segmentation,
    weighted_average_probabilities,
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


def test_class_aware_hd_postprocess_keeps_vessel_branches_and_removes_remote_island():
    seg = np.zeros((32, 32, 32), dtype=np.uint8)
    seg[4:14, 4:14, 4:14] = 1
    seg[14:17, 8:11, 8:11] = 7
    seg[17:20, 8:11, 8:11] = 7
    seg[20:23, 8:11, 8:11] = 7
    seg[28:31, 28:31, 28:31] = 7

    rules = build_class_aware_hd_rules(labels=[1, 2, 3, 4, 5, 6, 7], max_distance_to_heart_mm=6)
    cleaned = postprocess_label_array(seg, labels=[1, 2, 3, 4, 5, 6, 7], class_rules=rules)

    assert np.count_nonzero(cleaned[14:17, 8:11, 8:11] == 7) == 27
    assert np.count_nonzero(cleaned[17:20, 8:11, 8:11] == 7) == 27
    assert np.count_nonzero(cleaned[20:23, 8:11, 8:11] == 7) == 27
    assert np.count_nonzero(cleaned[28:31, 28:31, 28:31] == 7) == 0


def test_class_aware_hd_postprocess_fills_chamber_holes_without_overwriting_other_labels():
    seg = np.zeros((9, 9, 9), dtype=np.uint8)
    seg[1:8, 1:8, 1:8] = 1
    seg[4, 4, 4] = 0
    seg[4, 4, 5] = 5

    rules = build_class_aware_hd_rules(labels=[1, 2, 3, 4, 5, 6, 7])
    cleaned = postprocess_label_array(seg, labels=[1, 2, 3, 4, 5, 6, 7], class_rules=rules)

    assert cleaned[4, 4, 4] == 1
    assert cleaned[4, 4, 5] == 5


def test_ct_hu_window_maps_to_nnunet_normalized_space():
    lower, upper = ct_hu_window_to_normalized(
        lower_hu=0,
        upper_hu=900,
        intensity_properties={"mean": 100, "std": 50},
    )

    assert lower == -2
    assert upper == 16


def test_ct_window_rescale_clips_and_expands_selected_contrast_range():
    data = torch.tensor([[[[[-2.0, -1.0, 0.0, 1.0, 2.0]]]]])

    output = apply_ct_window_rescale_torch(data, lower=-1.0, upper=1.0, blend=1.0)

    expected = torch.tensor([[[[[-2.0, -2.0, 0.0, 2.0, 2.0]]]]])
    assert torch.allclose(output, expected)


def test_prediction_sanity_accepts_valid_official_labels():
    pred = np.array([0, 500, 600, 420, 550, 205, 820, 850], dtype=np.uint16)

    issues = check_array_labels(
        pred,
        allowed_labels={0, 500, 600, 420, 550, 205, 820, 850},
        required_labels={500, 600, 420, 550, 205, 820, 850},
        path="Case001_label.nii.gz",
    )

    assert issues == []


def test_prediction_sanity_reports_unexpected_label_values():
    pred = np.array([0, 500, 999], dtype=np.uint16)

    issues = check_array_labels(
        pred,
        allowed_labels={0, 500, 600, 420, 550, 205, 820, 850},
        required_labels=set(),
        path="Case001_label.nii.gz",
    )

    assert [issue.code for issue in issues] == ["unexpected_label"]
    assert "999" in issues[0].message


def test_prediction_sanity_reports_geometry_mismatches():
    input_geometry = GeometrySignature(
        size=(8, 7, 6),
        spacing=(1.0, 1.0, 2.0),
        origin=(0.0, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    )
    pred_geometry = GeometrySignature(
        size=(8, 7, 5),
        spacing=(1.0, 1.0, 2.5),
        origin=(0.0, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    )

    issues = compare_geometry(input_geometry, pred_geometry, path="Case001_label.nii.gz")

    assert [issue.code for issue in issues] == ["shape_mismatch", "spacing_mismatch"]


def test_compare_metric_summaries_reports_experiment_delta_against_baseline():
    rows = [
        {"run": "ct-baseline-legacy", "fold": "0", "class": "AO", "n_cases": "2", "dsc": "0.90", "hd_mm": "8.0", "assd_mm": "1.2"},
        {"run": "ct-aopa-adaptive", "fold": "0", "class": "AO", "n_cases": "2", "dsc": "0.91", "hd_mm": "7.5", "assd_mm": "1.1"},
    ]

    compared = compare_summary_rows(rows, baseline_run="ct-baseline-legacy", include_folds=True)

    assert compared[0]["run"] == "ct-aopa-adaptive"
    assert compared[0]["fold"] == "0"
    assert compared[0]["class"] == "AO"
    assert compared[0]["dsc_delta"] == 0.01
    assert compared[0]["hd_mm_delta"] == -0.5
    assert compared[0]["assd_mm_delta"] == -0.1


def test_compare_metric_summaries_adds_mean_rows_across_folds():
    rows = [
        {"run": "mr-baseline-class-aware-hd", "fold": "0", "class": "PA", "n_cases": "2", "dsc": "0.80", "hd_mm": "9.0", "assd_mm": "1.5"},
        {"run": "mr-baseline-class-aware-hd", "fold": "1", "class": "PA", "n_cases": "2", "dsc": "0.82", "hd_mm": "7.0", "assd_mm": "1.1"},
        {"run": "mr-ens-final-best-07-03", "fold": "0", "class": "PA", "n_cases": "2", "dsc": "0.81", "hd_mm": "8.0", "assd_mm": "1.4"},
        {"run": "mr-ens-final-best-07-03", "fold": "1", "class": "PA", "n_cases": "2", "dsc": "0.83", "hd_mm": "6.0", "assd_mm": "1.0"},
    ]

    compared = compare_summary_rows(rows, baseline_run="mr-baseline-class-aware-hd")

    assert len(compared) == 1
    assert compared[0]["fold"] == "mean"
    assert compared[0]["dsc"] == 0.82
    assert compared[0]["baseline_dsc"] == 0.81
    assert compared[0]["dsc_delta"] == 0.01
    assert compared[0]["hd_mm_delta"] == -1.0


def test_weighted_probability_ensemble_renormalizes_and_uses_argmax():
    prob_a = np.zeros((3, 2, 1, 1), dtype=np.float32)
    prob_b = np.zeros((3, 2, 1, 1), dtype=np.float32)
    prob_a[:, 0, 0, 0] = [0.1, 0.8, 0.1]
    prob_b[:, 0, 0, 0] = [0.1, 0.2, 0.7]
    prob_a[:, 1, 0, 0] = [0.2, 0.3, 0.5]
    prob_b[:, 1, 0, 0] = [0.2, 0.7, 0.1]

    ensembled = weighted_average_probabilities([prob_a, prob_b], weights=[0.75, 0.25])
    seg = argmax_segmentation(ensembled)

    assert np.allclose(ensembled.sum(axis=0), 1.0)
    assert seg.tolist() == [[[1]], [[1]]]


def test_ct_aopa_adaptive_threshold_clamps_from_seed_median_prob():
    class_prob = np.array([0.20, 0.60, 0.90], dtype=np.float32)

    assert adaptive_seed_threshold(class_prob, np.array([False, True, False])) == 0.27
    assert adaptive_seed_threshold(np.array([0.95, 0.90], dtype=np.float32), np.array([True, True])) == 0.30
    assert adaptive_seed_threshold(np.array([0.30, 0.40], dtype=np.float32), np.array([True, True])) == 0.18


def test_ct_aopa_hysteresis_grows_seed_connected_vessel_without_remote_or_high_conf_other():
    pred = np.zeros((9, 9, 9), dtype=np.uint8)
    pred[3:5, 3:5, 3:5] = 6
    prob = np.zeros((8, 9, 9, 9), dtype=np.float32)
    prob[6, pred == 6] = 0.80
    prob[6, 5, 4, 4] = 0.26
    prob[6, 6, 4, 4] = 0.27
    prob[6, 0, 0, 0] = 0.90
    prob[6, 4, 5, 4] = 0.90
    prob[1, 4, 5, 4] = 0.70

    refined, stats = vessel_hysteresis_growing(
        pred,
        prob,
        class_idx=6,
        spacing=(1.0, 1.0, 1.0),
        p_class_threshold=0.25,
        threshold_mode="fixed",
        bbox_margin_mm=5,
        max_distance_mm=5,
    )

    assert stats.threshold == 0.25
    assert refined[5, 4, 4] == 6
    assert refined[6, 4, 4] == 6
    assert refined[0, 0, 0] == 0
    assert refined[4, 5, 4] == 0


def test_ct_aopa_adaptive_hysteresis_recovers_low_confidence_connected_gap():
    pred = np.zeros((7, 7, 7), dtype=np.uint8)
    pred[2:4, 2:4, 2:4] = 7
    prob = np.zeros((8, 7, 7, 7), dtype=np.float32)
    prob[7, pred == 7] = 0.40
    prob[7, 4, 3, 3] = 0.20

    fixed, _ = vessel_hysteresis_growing(
        pred,
        prob,
        class_idx=7,
        spacing=(1.0, 1.0, 1.0),
        p_class_threshold=0.25,
        threshold_mode="fixed",
    )
    adaptive, stats = vessel_hysteresis_growing(
        pred,
        prob,
        class_idx=7,
        spacing=(1.0, 1.0, 1.0),
        threshold_mode="adaptive",
    )

    assert fixed[4, 3, 3] == 0
    assert stats.threshold == 0.18
    assert adaptive[4, 3, 3] == 7


def test_ct_aopa_component_score_cleanup_removes_independent_low_confidence_component():
    pred = np.zeros((9, 9, 9), dtype=np.uint8)
    pred[2:4, 2:4, 2:4] = 6
    pred[6:8, 6:8, 6:8] = 6
    prob = np.zeros((8, 9, 9, 9), dtype=np.float32)
    prob[0] = 0.05
    prob[6, 2:4, 2:4, 2:4] = 0.85
    prob[6, 6:8, 6:8, 6:8] = 0.10
    seed = pred == 6
    seed[6:8, 6:8, 6:8] = False

    cleaned, stats = component_score_cleanup(
        pred,
        prob,
        class_idx=6,
        spacing=(1.0, 1.0, 1.0),
        seed_mask=seed,
        min_voxels=2,
        min_mean_prob=0.35,
        min_p10_prob=0.15,
    )

    assert stats.components_removed == 1
    assert np.count_nonzero(cleaned[2:4, 2:4, 2:4] == 6) == 8
    assert np.count_nonzero(cleaned[6:8, 6:8, 6:8] == 6) == 0


def test_dice_per_label_reports_class_level_failures():
    pred = np.array([1, 1, 2, 2, 0, 0, 3, 0])
    ref = np.array([1, 1, 2, 0, 2, 0, 0, 3])

    scores = dice_per_label(pred, ref, labels=[1, 2, 3])

    assert scores[1] == 1.0
    assert scores[2] == 0.5
    assert scores[3] == 0.0
    assert worst_class(scores, LABEL_NAMES) == ("LA", 0.0)


def test_segmentation_metrics_are_zero_for_identical_masks():
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[1:4, 1:4, 1:4] = True

    metrics = compute_binary_metrics(mask, mask, spacing_xyz=(1.0, 1.0, 1.0))

    assert metrics.dsc == 1.0
    assert metrics.hd_mm == 0.0
    assert metrics.assd_mm == 0.0


def test_segmentation_metrics_respect_physical_spacing():
    pred = np.zeros((5, 5, 5), dtype=bool)
    ref = np.zeros((5, 5, 5), dtype=bool)
    pred[2, 2, 2] = True
    ref[3, 2, 2] = True

    metrics = compute_binary_metrics(pred, ref, spacing_xyz=(1.0, 1.0, 2.0))

    assert metrics.dsc == 0.0
    assert metrics.hd_mm == 2.0
    assert metrics.assd_mm == 2.0


def test_segmentation_metric_crop_preserves_union_foreground_with_padding():
    pred = np.zeros((20, 20, 20), dtype=bool)
    ref = np.zeros((20, 20, 20), dtype=bool)
    pred[10, 10, 10] = True
    ref[12, 10, 10] = True

    cropped_pred, cropped_ref = crop_to_union_foreground(pred, ref, padding=2)

    assert cropped_pred.shape == (7, 5, 5)
    assert cropped_ref.shape == (7, 5, 5)
    assert cropped_pred.sum() == 1
    assert cropped_ref.sum() == 1
    assert compute_binary_metrics(pred, ref, spacing_xyz=(1.0, 1.0, 1.0)) == compute_binary_metrics(
        cropped_pred,
        cropped_ref,
        spacing_xyz=(1.0, 1.0, 1.0),
    )


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
        assert "unpack_dataset" not in [arg.arg for arg in init_method.args.args], (
            f"{name}.__init__ must not expose unpack_dataset because older nnU-Net "
            "base trainers reflect child init parameters"
        )
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


def test_wholeheart_trainer_allows_epoch_count_env_override():
    source = Path("track4_wholeheart/nnunet_extensions/nnUNetTrainerWholeHeartAug.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    trainer_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "nnUNetTrainerWholeHeartAug"
    )
    init_method = next(
        child for child in trainer_class.body if isinstance(child, ast.FunctionDef) and child.name == "__init__"
    )

    epoch_assignments = [
        node
        for node in ast.walk(init_method)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and target.attr == "num_epochs"
            for target in node.targets
        )
    ]

    assert any(
        isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "_env_int"
        and len(node.value.args) == 2
        and isinstance(node.value.args[0], ast.Constant)
        and node.value.args[0].value == "WHOLEHEART_NUM_EPOCHS"
        for node in epoch_assignments
    )


def test_wholeheart_trainer_exposes_ct_window_env_switches():
    source = Path("track4_wholeheart/nnunet_extensions/nnUNetTrainerWholeHeartAug.py").read_text(encoding="utf-8")

    assert "WHOLEHEART_CT_WINDOW_AUG" in source
    assert "WHOLEHEART_CT_WINDOW_PROB" in source
    assert "WHOLEHEART_CT_WINDOW_LOWER_RANGE" in source
    assert "WHOLEHEART_CT_WINDOW_UPPER_RANGE" in source
    assert "WHOLEHEART_CT_WINDOW_BLEND" in source


def test_predict_val_exposes_optional_sanity_check():
    source = Path("track4_wholeheart/scripts/predict_val.sh").read_text(encoding="utf-8")

    assert "SANITY_CHECK" in source
    assert "check_prediction_sanity.py" in source
    assert "--label-space official" in source


def test_predict_val_logs_submission_safeguards():
    source = Path("track4_wholeheart/scripts/predict_val.sh").read_text(encoding="utf-8")

    assert "expected_postprocess_preset=" in source
    assert "save_probabilities=enabled" in source
    assert "WARNING: ${MODALITY^^} prediction is not using full 5-fold ensemble" in source
    assert "WARNING: ${MODALITY^^} current best postprocess preset is" in source
    assert "CT current best postprocess preset is legacy" in source
    assert "MR current best postprocess preset is class-aware-hd" in source


def test_predict_val_exposes_ct_aopa_soft_patch_controls():
    source = Path("track4_wholeheart/scripts/predict_val.sh").read_text(encoding="utf-8")

    assert "CT_AOPA_SOFT" in source
    assert "postprocess_ct_aopa_soft.py" in source
    assert "--threshold-mode" in source
    assert "CT_AOPA_COMPONENT_SCORE" in source
    assert "CT_AOPA_ENABLE_CLOSING" in source
    assert "CT AOPA soft patch requires POSTPROCESS=1 and POSTPROCESS_PRESET=legacy" in source


def test_raw_unlabeled_pool_reuses_cached_patches(monkeypatch, tmp_path):
    for idx in range(3):
        (tmp_path / f"case_{idx:03d}_0000.nii.gz").touch()

    pool = WholeHeartRawUnlabeledPool(
        image_dir=tmp_path,
        patch_size=(2, 2, 2),
        modality="ct",
        cache_size=1,
        patch_cache_size=6,
    )
    reads = []

    def fake_read_image(path):
        reads.append(path)
        base = float(len(reads))
        return torch.full((4, 4, 4), base, dtype=torch.float32)

    monkeypatch.setattr(pool, "_read_image", fake_read_image)

    first = pool.sample_batch(batch_size=2, device=torch.device("cpu"), dtype=torch.float32)
    second = pool.sample_batch(batch_size=2, device=torch.device("cpu"), dtype=torch.float32)

    assert first.shape == (2, 1, 2, 2, 2)
    assert second.shape == (2, 1, 2, 2, 2)
    assert len(reads) == 3


def test_install_script_copies_unlabeled_pool_helper():
    source = Path("track4_wholeheart/scripts/install_wholeheart_trainer.py").read_text(encoding="utf-8")

    assert "unlabeled_pool.py" in source
    assert "ct_window_augmentation.py" in source


def test_predict_script_exposes_class_aware_postprocess_preset():
    source = Path("track4_wholeheart/scripts/predict_val.sh").read_text(encoding="utf-8")

    assert "POSTPROCESS_PRESET" in source
    assert "--preset" in source
    assert "WHOLEHEART_DISTANCE_MM" in source
    assert "VESSEL_MIN_COMPONENT_SIZE" in source


def test_medsam2_prompt_uses_largest_slice_bbox_with_padding():
    mask = np.zeros((4, 10, 12), dtype=bool)
    mask[1, 3:5, 4:6] = True
    mask[2, 2:7, 3:9] = True

    key_slice, bbox = central_bbox_prompt(mask, padding=2)

    assert key_slice == 2
    assert bbox.tolist() == [1.0, 0.0, 10.0, 8.0]


def test_medsam2_candidate_is_constrained_near_original_and_empty_regions():
    original = np.zeros((1, 7, 7), dtype=bool)
    original[0, 3, 3] = True
    medsam = np.zeros_like(original)
    medsam[0, 2:5, 2:5] = True
    medsam[0, 0, 0] = True
    current_seg = np.zeros((1, 7, 7), dtype=np.uint8)
    current_seg[0, 2, 2] = 4

    candidate = constrained_candidate(
        medsam_mask=medsam,
        original_mask=original,
        current_seg=current_seg,
        label=5,
        dilation_radius=1,
    )

    assert candidate[0, 3, 3]
    assert not candidate[0, 0, 0]
    assert not candidate[0, 2, 2]


def test_medsam2_acceptance_rejects_unstable_volume_or_overlap():
    original = np.zeros((1, 6, 6), dtype=bool)
    original[0, 2:4, 2:4] = True
    same = original.copy()
    too_large = np.zeros_like(original)
    too_large[0, 1:5, 1:5] = True
    shifted = np.zeros_like(original)
    shifted[0, 0:2, 0:2] = True

    assert should_accept_candidate(original, same, 0.8, 1.2, 0.5)[0]
    assert should_accept_candidate(original, too_large, 0.8, 1.2, 0.5)[:2] == (False, "too_large")
    assert should_accept_candidate(original, shifted, 0.8, 1.2, 0.5)[:2] == (False, "low_iou")


def test_medsam2_normalization_uses_nonzero_foreground_when_available():
    image = np.array([0.0, 10.0, 20.0, 30.0], dtype=np.float32).reshape(1, 2, 2)

    normalized = normalize_volume_to_uint8(image, lower_percentile=0, upper_percentile=100)

    assert normalized.tolist() == [[[0, 0], [128, 255]]]
