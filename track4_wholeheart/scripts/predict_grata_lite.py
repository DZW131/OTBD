from __future__ import annotations

import argparse
import json
import math
import os
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F


@dataclass
class AdaptStats:
    steps_attempted: int = 0
    steps_applied: int = 0
    mean_alignment: float = math.nan
    mean_entropy: float = math.nan
    mean_consistency: float = math.nan


def parse_int_list(value: str | None) -> list[int]:
    if value is None or not value.strip():
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def split_cases(splits_json: Path, fold: int) -> list[str]:
    with splits_json.open("r", encoding="utf-8") as f:
        splits = json.load(f)
    if fold < 0 or fold >= len(splits):
        raise ValueError(f"Fold {fold} is outside split file with {len(splits)} folds")
    cases = splits[fold].get("val", [])
    if not cases:
        raise ValueError(f"No validation cases found for fold {fold} in {splits_json}")
    return [str(case) for case in cases]


def center_group(case_id: str) -> str:
    if case_id.startswith("Case3"):
        return "MR_C/D"
    if case_id.startswith("Case5"):
        return "MR_E"
    return "unknown"


def limit_cases_by_center(cases: Sequence[str], limit: int | None) -> list[str]:
    if limit is None or limit <= 0:
        return list(cases)
    counts: dict[str, int] = {}
    selected: list[str] = []
    for case in cases:
        group = center_group(case)
        counts.setdefault(group, 0)
        if counts[group] >= limit:
            continue
        counts[group] += 1
        selected.append(case)
    return selected


def image_list_for_cases(input_dir: Path, cases: Sequence[str]) -> list[list[str]]:
    items: list[list[str]] = []
    for case in cases:
        candidates = [
            input_dir / f"{case}_0000.nii.gz",
            input_dir / f"{case}.nii.gz",
        ]
        for candidate in candidates:
            if candidate.exists():
                items.append([str(candidate)])
                break
        else:
            raise FileNotFoundError(f"Could not find image for case {case} under {input_dir}")
    return items


def output_truncated_for_cases(output_dir: Path, cases: Sequence[str]) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    return [str(output_dir / case) for case in cases]


def module_is_norm(module: torch.nn.Module) -> bool:
    return "norm" in module.__class__.__name__.lower()


def selected_trainable_parameters(
    network: torch.nn.Module,
    scope: str,
) -> list[torch.nn.Parameter]:
    for parameter in network.parameters():
        parameter.requires_grad_(False)

    selected: list[torch.nn.Parameter] = []
    if scope in {"norm-affine", "norm-and-decoder-last"}:
        for module in network.modules():
            if not module_is_norm(module):
                continue
            for name in ("weight", "bias"):
                parameter = getattr(module, name, None)
                if isinstance(parameter, torch.nn.Parameter):
                    parameter.requires_grad_(True)
                    selected.append(parameter)

    if scope in {"decoder-last", "norm-and-decoder-last"}:
        modules = list(network.modules())
        for module in reversed(modules):
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Conv3d)):
                for parameter in module.parameters(recurse=False):
                    parameter.requires_grad_(True)
                    selected.append(parameter)
                break

    # Keep object identity unique while preserving order.
    unique: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for parameter in selected:
        ident = id(parameter)
        if ident not in seen:
            seen.add(ident)
            unique.append(parameter)
    return unique


def center_crop_or_pad(data: torch.Tensor, patch_size: Sequence[int]) -> torch.Tensor:
    if data.ndim != 4:
        raise ValueError(f"Expected preprocessed data with shape [C, Z, Y, X], got {tuple(data.shape)}")
    patch = tuple(int(v) for v in patch_size)
    spatial = tuple(int(v) for v in data.shape[1:])
    slices = [slice(None)]
    pads_reversed: list[int] = []
    for size, target in zip(reversed(spatial), reversed(patch)):
        if size >= target:
            start = (size - target) // 2
            pads_reversed.extend([0, 0])
            slices.insert(1, slice(start, start + target))
        else:
            total = target - size
            left = total // 2
            right = total - left
            pads_reversed.extend([left, right])
            slices.insert(1, slice(None))

    cropped = data[tuple(slices)]
    if any(pads_reversed):
        cropped = F.pad(cropped, pads_reversed, mode="constant", value=0.0)
    return cropped


def intensity_strong_aug(
    patch: torch.Tensor,
    noise_std: float,
    brightness_std: float,
) -> torch.Tensor:
    out = patch
    if brightness_std > 0:
        scale = 1.0 + torch.randn((), device=patch.device, dtype=patch.dtype) * float(brightness_std)
        out = out * scale
    if noise_std > 0:
        std = torch.clamp(out.detach().std(), min=torch.tensor(1e-3, device=out.device, dtype=out.dtype))
        out = out + torch.randn_like(out) * std * float(noise_std)
    return out


def tensor_list_dot(left: Sequence[torch.Tensor | None], right: Sequence[torch.Tensor | None]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dot = None
    left_norm = None
    right_norm = None
    for lval, rval in zip(left, right):
        if lval is None or rval is None:
            continue
        lflat = lval.detach().flatten().float()
        rflat = rval.detach().flatten().float()
        item_dot = torch.dot(lflat, rflat)
        item_left = torch.dot(lflat, lflat)
        item_right = torch.dot(rflat, rflat)
        dot = item_dot if dot is None else dot + item_dot
        left_norm = item_left if left_norm is None else left_norm + item_left
        right_norm = item_right if right_norm is None else right_norm + item_right
    if dot is None or left_norm is None or right_norm is None:
        zero = torch.tensor(0.0)
        return zero, zero, zero
    return dot, left_norm, right_norm


def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    prob = torch.softmax(logits.float(), dim=1)
    return -(prob * torch.log(torch.clamp(prob, min=1e-8))).sum(dim=1).mean()


def first_output(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


class GraTaLiteAdapter:
    def __init__(self, args: argparse.Namespace):
        self.steps = int(args.steps)
        self.lr = float(args.lr)
        self.param_scope = args.param_scope
        self.entropy_weight = float(args.entropy_weight)
        self.consistency_weight = float(args.consistency_weight)
        self.alignment_threshold = float(args.alignment_threshold)
        self.noise_std = float(args.noise_std)
        self.brightness_std = float(args.brightness_std)
        self.max_grad_norm = float(args.max_grad_norm)

    def adapt(self, network: torch.nn.Module, data: torch.Tensor, patch_size: Sequence[int], device: torch.device) -> AdaptStats:
        if self.steps <= 0:
            return AdaptStats()

        network.to(device)
        network.eval()
        trainable = selected_trainable_parameters(network, self.param_scope)
        if not trainable:
            print(f"GraTa-lite: no trainable parameters selected for scope={self.param_scope}; skipping", flush=True)
            return AdaptStats()

        optimizer = torch.optim.AdamW(trainable, lr=self.lr, weight_decay=0.0)
        patch = center_crop_or_pad(data.float(), patch_size).unsqueeze(0).to(device)
        alignments: list[float] = []
        entropies: list[float] = []
        consistencies: list[float] = []
        applied = 0

        for _ in range(self.steps):
            weak = patch
            strong = intensity_strong_aug(patch, self.noise_std, self.brightness_std)
            weak_logits = first_output(network(weak))
            strong_logits = first_output(network(strong))
            weak_prob = torch.softmax(weak_logits.float(), dim=1)
            strong_prob = torch.softmax(strong_logits.float(), dim=1)
            entropy_loss = softmax_entropy(weak_logits)
            consistency_loss = F.mse_loss(strong_prob, weak_prob.detach())

            entropy_grads = torch.autograd.grad(
                entropy_loss,
                trainable,
                retain_graph=True,
                allow_unused=True,
            )
            consistency_grads = torch.autograd.grad(
                consistency_loss,
                trainable,
                retain_graph=True,
                allow_unused=True,
            )
            dot, entropy_norm, consistency_norm = tensor_list_dot(entropy_grads, consistency_grads)
            denom = torch.sqrt(entropy_norm * consistency_norm).clamp_min(1e-12)
            alignment = float((dot / denom).cpu()) if float(denom.cpu()) > 0 else 0.0
            alignments.append(alignment)
            entropies.append(float(entropy_loss.detach().cpu()))
            consistencies.append(float(consistency_loss.detach().cpu()))

            if alignment < self.alignment_threshold:
                optimizer.zero_grad(set_to_none=True)
                continue

            loss = self.entropy_weight * entropy_loss + self.consistency_weight * consistency_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable, self.max_grad_norm)
            optimizer.step()
            applied += 1

        return AdaptStats(
            steps_attempted=self.steps,
            steps_applied=applied,
            mean_alignment=float(sum(alignments) / len(alignments)) if alignments else math.nan,
            mean_entropy=float(sum(entropies) / len(entropies)) if entropies else math.nan,
            mean_consistency=float(sum(consistencies) / len(consistencies)) if consistencies else math.nan,
        )


def patch_predictor_with_grata(predictor, adapter: GraTaLiteAdapter) -> None:
    import torch
    from nnunetv2.configuration import default_num_processes
    from nnunetv2.utilities.helpers import empty_cache
    try:
        from torch._dynamo import OptimizedModule
    except Exception:  # pragma: no cover
        OptimizedModule = tuple()  # type: ignore

    def predict_logits_from_preprocessed_data_with_grata(self, data: torch.Tensor) -> torch.Tensor:
        n_threads = torch.get_num_threads()
        torch.set_num_threads(default_num_processes if default_num_processes < n_threads else n_threads)
        prediction = None

        for params in self.list_of_parameters:
            if not isinstance(self.network, OptimizedModule):
                self.network.load_state_dict(params)
                network_for_adapt = self.network
            else:
                self.network._orig_mod.load_state_dict(params)
                network_for_adapt = self.network._orig_mod

            stats = adapter.adapt(network_for_adapt, data, self.configuration_manager.patch_size, self.device)
            print(
                "GraTa-lite stats: "
                f"attempted={stats.steps_attempted} applied={stats.steps_applied} "
                f"alignment={stats.mean_alignment:.4f} entropy={stats.mean_entropy:.4f} "
                f"consistency={stats.mean_consistency:.6f}",
                flush=True,
            )

            if prediction is None:
                prediction = self.predict_sliding_window_return_logits(data).to("cpu")
            else:
                prediction += self.predict_sliding_window_return_logits(data).to("cpu")

        if len(self.list_of_parameters) > 1:
            prediction /= len(self.list_of_parameters)

        torch.set_num_threads(n_threads)
        empty_cache(self.device)
        return prediction

    predictor.predict_logits_from_preprocessed_data = types.MethodType(
        predict_logits_from_preprocessed_data_with_grata,
        predictor,
    )


def initialize_predictor_from_trained_model_folder_compat(
    predictor,
    model_dir: Path,
    folds: Sequence[int | str],
    checkpoint_name: str,
) -> None:
    try:
        predictor.initialize_from_trained_model_folder(
            str(model_dir),
            list(folds),
            checkpoint_name=checkpoint_name,
        )
        return
    except TypeError as exc:
        message = str(exc)
        if "build_network_architecture" not in message or "num_output_channels" not in message:
            raise

    print(
        "Standard predictor initialization failed because the trainer build_network_architecture "
        "signature differs; retrying with MAE/nnSSL-compatible initialization.",
        flush=True,
    )

    import nnunetv2
    from batchgenerators.utilities.file_and_folder_operations import join, load_json
    from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    dataset_json = load_json(join(str(model_dir), "dataset.json"))
    plans = load_json(join(str(model_dir), "plans.json"))
    plans_manager = PlansManager(plans)

    parameters = []
    trainer_name = None
    configuration_name = None
    inference_allowed_mirroring_axes = None
    for i, fold in enumerate(folds):
        checkpoint = torch.load(
            join(str(model_dir), f"fold_{fold}", checkpoint_name),
            map_location=torch.device("cpu"),
            weights_only=False,
        )
        if i == 0:
            trainer_name = checkpoint["trainer_name"]
            configuration_name = checkpoint["init_args"]["configuration"]
            inference_allowed_mirroring_axes = checkpoint.get("inference_allowed_mirroring_axes")
        parameters.append(checkpoint["network_weights"])

    if trainer_name is None or configuration_name is None:
        raise RuntimeError(f"No checkpoints loaded from {model_dir} folds={folds}")

    configuration_manager = plans_manager.get_configuration(configuration_name)
    label_manager = plans_manager.get_label_manager(dataset_json)
    num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
    trainer_class = recursive_find_python_class(
        join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
        trainer_name,
        "nnunetv2.training.nnUNetTrainer",
    )
    if trainer_class is None:
        raise RuntimeError(f"Unable to locate trainer class {trainer_name}")

    try:
        network = trainer_class.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            configuration_manager.patch_size,
            num_input_channels,
            label_manager.num_segmentation_heads,
            enable_deep_supervision=False,
        )
    except TypeError:
        network = trainer_class.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            label_manager.num_segmentation_heads,
            enable_deep_supervision=False,
        )
    network.load_state_dict(parameters[0])
    predictor.manual_initialization(
        network=network,
        plans_manager=plans_manager,
        configuration_manager=configuration_manager,
        parameters=parameters,
        dataset_json=dataset_json,
        trainer_name=trainer_name,
        inference_allowed_mirroring_axes=inference_allowed_mirroring_axes,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Conservative GraTa-style single-case TTA wrapper for nnU-Net validation experiments."
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", nargs="+", required=True, help="One or more folds to load, for example 0.")
    parser.add_argument("--checkpoint", default="checkpoint_final.pth")
    parser.add_argument("--splits-json", type=Path, default=None)
    parser.add_argument("--use-split-val", action="store_true")
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--max-cases-per-center", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tile-step-size", type=float, default=0.5)
    parser.add_argument("--disable-mirroring", action="store_true")
    parser.add_argument("--save-probabilities", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num-processes-preprocessing", type=int, default=3)
    parser.add_argument("--num-processes-export", type=int, default=3)

    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--param-scope", choices=["norm-affine", "decoder-last", "norm-and-decoder-last"], default="norm-affine")
    parser.add_argument("--entropy-weight", type=float, default=1.0)
    parser.add_argument("--consistency-weight", type=float, default=1.0)
    parser.add_argument("--alignment-threshold", type=float, default=0.0)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--brightness-std", type=float, default=0.02)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("nnUNet_compile", "0")

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    folds = [int(fold) if str(fold) != "all" else "all" for fold in args.folds]
    if args.use_split_val:
        if args.splits_json is None:
            raise SystemExit("--use-split-val requires --splits-json")
        if len(folds) != 1 or folds[0] == "all":
            raise SystemExit("--use-split-val currently expects exactly one numeric fold")
        cases = split_cases(args.splits_json, int(folds[0]))
    elif args.cases:
        cases = list(args.cases)
    else:
        cases = []

    if cases:
        if args.max_cases_per_center is not None:
            cases = limit_cases_by_center(cases, args.max_cases_per_center)
        if args.max_cases is not None:
            cases = cases[: args.max_cases]
        inputs = image_list_for_cases(args.input_dir, cases)
        outputs = output_truncated_for_cases(args.output_dir, cases)
        print(f"GraTa-lite cases ({len(cases)}): {', '.join(cases)}", flush=True)
    else:
        inputs = str(args.input_dir)
        outputs = str(args.output_dir)
        print("GraTa-lite cases: all images in input directory", flush=True)

    predictor = nnUNetPredictor(
        tile_step_size=args.tile_step_size,
        use_gaussian=True,
        use_mirroring=not args.disable_mirroring,
        perform_everything_on_device=True,
        device=torch.device(args.device),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    initialize_predictor_from_trained_model_folder_compat(predictor, args.model_dir, folds, args.checkpoint)
    patch_predictor_with_grata(predictor, GraTaLiteAdapter(args))
    predictor.predict_from_files(
        inputs,
        outputs,
        save_probabilities=args.save_probabilities,
        overwrite=args.overwrite,
        num_processes_preprocessing=args.num_processes_preprocessing,
        num_processes_segmentation_export=args.num_processes_export,
    )
    print(f"GraTa-lite predictions written to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
