"""Lazy-imported float64 CUDA fitting and scoring for directional GLR-CUSUM."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from .glr import (
    FAULT_IDS,
    NORMAL_ID,
    PERSISTENCE_WINDOW,
    ROBUST_SCALE_FACTOR,
    TRAIN_START,
    TRAIN_STOP,
    Z_CLIP,
    DirectionalGLRModel,
    DirectionalGLRPath,
    compare_directional_paths,
    normalize_kappa,
    score_directional_glr_numpy,
    validate_feature_mapping,
    validate_labeled_feature_mapping,
)


@dataclass(frozen=True)
class DirectionalGLRFoldFit:
    """One ridge candidate trained with one grouped fold held out."""

    ridge_lambda: float
    held_out_fold: int | None
    training_run_names: tuple[str, ...]
    held_out_run_names: tuple[str, ...]
    model: DirectionalGLRModel
    convergence: dict[str, object]


@dataclass(frozen=True)
class DirectionalGLRGPUResult:
    """Named independent paths and explicit CUDA execution provenance."""

    by_run: dict[str, DirectionalGLRPath]
    execution: dict[str, object]
    runtime_environment: dict[str, object]
    sequence_length: int
    feature_count: int


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _positive_float(value: Any, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return result


def _torch_and_device(device: str) -> tuple[Any, Any, dict[str, object]]:
    try:
        import torch
    except (ImportError, OSError) as error:
        raise RuntimeError("a CUDA-enabled PyTorch installation is required") from error
    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device!r} is not available")
    properties = torch.cuda.get_device_properties(target)
    runtime = {
        "backend": "pytorch_cuda_directional_glr",
        "requested_device": str(device),
        "resolved_device": str(target),
        "device_name": str(torch.cuda.get_device_name(target)),
        "compute_capability": [int(properties.major), int(properties.minor)],
        "total_memory_bytes": int(properties.total_memory),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "cuda_available": bool(torch.cuda.is_available()),
        "dtype": "float64",
        "tf32": False,
    }
    return torch, target, runtime


@contextmanager
def _tf32_disabled(torch: Any) -> Iterator[None]:
    previous_matmul = bool(torch.backends.cuda.matmul.allow_tf32)
    previous_cudnn = bool(torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul
        torch.backends.cudnn.allow_tf32 = previous_cudnn


def _class_training_arrays(
    records: tuple[tuple[str, int, np.ndarray], ...],
) -> dict[int, np.ndarray]:
    return {
        label: np.stack(
            [
                matrix[TRAIN_START:TRAIN_STOP]
                for _, run_label, matrix in records
                if run_label == label
            ],
            axis=0,
        )
        for label in (NORMAL_ID, *FAULT_IDS)
    }


def fit_directional_glr_cuda(
    features_by_run: Mapping[str, np.ndarray],
    labels_by_run: Mapping[str, int],
    *,
    ridge_lambda: float,
    kappa: float | Sequence[float] | np.ndarray = 0.0,
    device: str = "cuda",
    max_iter: int = 100,
    max_eval: int | None = None,
    tolerance_grad: float = 1e-9,
    tolerance_change: float = 1e-12,
    history_size: int = 50,
    require_convergence: bool = False,
) -> tuple[DirectionalGLRModel, dict[str, object]]:
    """Jointly fit three one-vs-normal ridge-logistic heads with CUDA LBFGS.

    Every head receives half its data loss from normal runs and half from its
    target fault. A time mean is taken inside each run before the run mean, so
    neither class nor an individual run can dominate the objective.
    """
    records = validate_labeled_feature_mapping(
        features_by_run,
        labels_by_run,
        require_balanced=True,
    )
    ridge_lambda = _positive_float(ridge_lambda, name="ridge_lambda")
    kappa_array = normalize_kappa(kappa)
    max_iter = _positive_integer(max_iter, name="max_iter")
    history_size = _positive_integer(history_size, name="history_size")
    if max_eval is None:
        max_eval = max(1, int(max_iter * 5 // 4))
    max_eval = _positive_integer(max_eval, name="max_eval")
    tolerance_grad = _positive_float(tolerance_grad, name="tolerance_grad")
    tolerance_change = _positive_float(
        tolerance_change,
        name="tolerance_change",
    )
    if not isinstance(require_convergence, (bool, np.bool_)):
        raise ValueError("require_convergence must be boolean")

    arrays = _class_training_arrays(records)
    feature_count = int(next(iter(arrays.values())).shape[2])
    torch, target, runtime = _torch_and_device(device)
    tensors = {
        label: torch.as_tensor(array, dtype=torch.float64, device=target)
        for label, array in arrays.items()
    }
    weights = torch.zeros(
        (len(FAULT_IDS), feature_count),
        dtype=torch.float64,
        device=target,
        requires_grad=True,
    )
    bias = torch.zeros(
        len(FAULT_IDS),
        dtype=torch.float64,
        device=target,
        requires_grad=True,
    )

    def head_losses() -> list[Any]:
        normal_logits = tensors[NORMAL_ID] @ weights.T + bias
        losses: list[Any] = []
        for head, fault_id in enumerate(FAULT_IDS):
            negative_by_run = torch.nn.functional.softplus(
                normal_logits[:, :, head]
            ).mean(dim=1)
            positive_logit = tensors[fault_id] @ weights[head] + bias[head]
            positive_by_run = torch.nn.functional.softplus(-positive_logit).mean(
                dim=1
            )
            data_loss = 0.5 * (
                negative_by_run.mean() + positive_by_run.mean()
            )
            penalty = 0.5 * ridge_lambda * torch.sum(weights[head] ** 2)
            losses.append(data_loss + penalty)
        return losses

    def objective() -> Any:
        return torch.stack(head_losses()).mean()

    optimizer = torch.optim.LBFGS(
        [weights, bias],
        lr=1.0,
        max_iter=max_iter,
        max_eval=max_eval,
        tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change,
        history_size=history_size,
        line_search_fn="strong_wolfe",
    )
    closure_evaluations = 0

    def closure() -> Any:
        nonlocal closure_evaluations
        closure_evaluations += 1
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("ridge-logistic objective became non-finite")
        loss.backward()
        return loss

    torch.cuda.reset_peak_memory_stats(target)
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with _tf32_disabled(torch):
        with torch.no_grad():
            initial_objective = float(objective().detach().cpu())
        begin.record()
        optimizer.step(closure)
        end.record()
        end.synchronize()
        elapsed_ms = float(begin.elapsed_time(end))
        final_loss = objective()
        gradients = torch.autograd.grad(final_loss, (weights, bias))
        gradient_inf_norm = max(
            float(torch.max(torch.abs(gradient)).detach().cpu())
            for gradient in gradients
        )
        final_objective = float(final_loss.detach().cpu())
        final_head_objectives = [
            float(value.detach().cpu()) for value in head_losses()
        ]
        normal_logits = tensors[NORMAL_ID] @ weights.T + bias
        flat_normal_logits = normal_logits.reshape(-1, len(FAULT_IDS))
        normal_center = torch.quantile(flat_normal_logits, 0.5, dim=0)
        absolute_deviation = torch.abs(flat_normal_logits - normal_center)
        normal_mad = torch.quantile(absolute_deviation, 0.5, dim=0)
        normal_scale = ROBUST_SCALE_FACTOR * normal_mad
        scale_fallback = (~torch.isfinite(normal_scale)) | (normal_scale <= 0.0)
        if bool(torch.any(scale_fallback)):
            sample_sd = torch.std(flat_normal_logits, dim=0, correction=1)
            normal_scale = torch.where(scale_fallback, sample_sd, normal_scale)
        if not bool(torch.isfinite(normal_scale).all()) or bool(
            torch.any(normal_scale <= 0.0)
        ):
            raise FloatingPointError("normal logit scale remains degenerate")

    state = optimizer.state.get(weights, {})
    optimizer_iterations = int(state.get("n_iter", 0))
    optimizer_evaluations = int(state.get("func_evals", closure_evaluations))
    if gradient_inf_norm <= tolerance_grad:
        termination_reason = "gradient_tolerance"
        converged = True
    elif optimizer_iterations >= max_iter:
        termination_reason = "maximum_iterations"
        converged = False
    elif optimizer_evaluations >= max_eval:
        termination_reason = "maximum_function_evaluations"
        converged = False
    else:
        termination_reason = "optimizer_change_tolerance"
        converged = True
    if bool(require_convergence) and not converged:
        raise RuntimeError(
            "CUDA LBFGS did not converge: "
            f"reason={termination_reason}, gradient_inf_norm={gradient_inf_norm:.3e}"
        )

    model = DirectionalGLRModel(
        fault_ids=FAULT_IDS,
        weights=weights.detach().cpu().numpy(),
        bias=bias.detach().cpu().numpy(),
        normal_logit_center=normal_center.detach().cpu().numpy(),
        normal_logit_scale=normal_scale.detach().cpu().numpy(),
        kappa=kappa_array,
        threshold=None,
    )
    counts = {
        str(label): sum(run_label == label for _, run_label, _ in records)
        for label in (NORMAL_ID, *FAULT_IDS)
    }
    metadata: dict[str, object] = {
        "backend": "pytorch_cuda_joint_three_head_lbfgs",
        "device": str(target),
        "device_name": runtime["device_name"],
        "dtype": "float64",
        "tf32": False,
        "fault_ids": list(FAULT_IDS),
        "normal_id": NORMAL_ID,
        "feature_count": feature_count,
        "training_time_slice_zero_based_half_open": [TRAIN_START, TRAIN_STOP],
        "time_samples_per_run": TRAIN_STOP - TRAIN_START,
        "training_run_count_by_label": counts,
        "loss_weighting": "equal_heads_equal_classes_equal_runs_equal_times",
        "ridge_lambda": ridge_lambda,
        "ridge_penalty": "lambda_over_2_times_squared_weight_norm_per_head",
        "bias_penalized": False,
        "initial_objective": initial_objective,
        "final_objective": final_objective,
        "final_head_objectives": final_head_objectives,
        "optimizer": "torch.optim.LBFGS_strong_wolfe",
        "initialization": "all_zero_no_randomness",
        "training_order": "run_names_sorted_lexicographically",
        "determinism_scope": "fixed_device_fixed_order_no_random_operations",
        "max_iter": max_iter,
        "max_eval": max_eval,
        "history_size": history_size,
        "tolerance_grad": tolerance_grad,
        "tolerance_change": tolerance_change,
        "optimizer_iterations": optimizer_iterations,
        "closure_evaluations": closure_evaluations,
        "optimizer_function_evaluations": optimizer_evaluations,
        "gradient_inf_norm": gradient_inf_norm,
        "termination_reason": termination_reason,
        "converged": converged,
        "cuda_event_elapsed_seconds": elapsed_ms / 1000.0,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(target)),
        "normal_logit_location": "training_normal_pooled_median",
        "normal_logit_scale": "1.4826_MAD_with_sample_SD_fallback",
        "normal_scale_fallback": [bool(value) for value in scale_fallback.cpu()],
        "runtime_environment": runtime,
    }
    del tensors, weights, bias
    return model, metadata


def _ridge_grid(values: Sequence[float]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("ridge_lambdas must be a nonempty sequence")
    normalized = tuple(_positive_float(value, name="ridge lambda") for value in values)
    if not normalized or len(normalized) != len(set(normalized)):
        raise ValueError("ridge_lambdas must be nonempty and unique")
    return normalized


def _fold_mapping(
    run_names: set[str],
    fold_by_run: Mapping[str, int],
) -> tuple[tuple[str, int], ...]:
    if not isinstance(fold_by_run, Mapping) or set(fold_by_run) != run_names:
        raise ValueError("fold_by_run must have exactly the training run names")
    records: list[tuple[str, int]] = []
    for name in sorted(run_names):
        value = fold_by_run[name]
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value,
            (int, np.integer),
        ):
            raise ValueError("fold identifiers must be nonnegative integers")
        fold = int(value)
        if fold < 0:
            raise ValueError("fold identifiers must be nonnegative integers")
        records.append((name, fold))
    if len({fold for _, fold in records}) < 2:
        raise ValueError("grouped fitting requires at least two folds")
    return tuple(records)


def fit_directional_glr_grid_cuda(
    features_by_run: Mapping[str, np.ndarray],
    labels_by_run: Mapping[str, int],
    *,
    ridge_lambdas: Sequence[float],
    fold_by_run: Mapping[str, int] | None = None,
    kappa: float | Sequence[float] | np.ndarray = 0.0,
    device: str = "cuda",
    max_iter: int = 100,
    max_eval: int | None = None,
    tolerance_grad: float = 1e-9,
    tolerance_change: float = 1e-12,
    history_size: int = 50,
    require_convergence: bool = False,
) -> tuple[DirectionalGLRFoldFit, ...]:
    """Fit every ridge candidate either fully or once per held-out group fold."""
    validate_labeled_feature_mapping(
        features_by_run,
        labels_by_run,
        require_balanced=True,
    )
    lambdas = _ridge_grid(ridge_lambdas)
    common = {
        "kappa": kappa,
        "device": device,
        "max_iter": max_iter,
        "max_eval": max_eval,
        "tolerance_grad": tolerance_grad,
        "tolerance_change": tolerance_change,
        "history_size": history_size,
        "require_convergence": require_convergence,
    }
    if fold_by_run is None:
        fits: list[DirectionalGLRFoldFit] = []
        names = tuple(sorted(features_by_run))
        for ridge_lambda in lambdas:
            model, convergence = fit_directional_glr_cuda(
                features_by_run,
                labels_by_run,
                ridge_lambda=ridge_lambda,
                **common,
            )
            fits.append(
                DirectionalGLRFoldFit(
                    ridge_lambda=ridge_lambda,
                    held_out_fold=None,
                    training_run_names=names,
                    held_out_run_names=(),
                    model=model,
                    convergence=convergence,
                )
            )
        return tuple(fits)

    fold_records = _fold_mapping(set(features_by_run), fold_by_run)
    fold_values = sorted({fold for _, fold in fold_records})
    fits = []
    for ridge_lambda in lambdas:
        for held_out_fold in fold_values:
            training_names = tuple(
                name for name, fold in fold_records if fold != held_out_fold
            )
            held_out_names = tuple(
                name for name, fold in fold_records if fold == held_out_fold
            )
            training_features = {name: features_by_run[name] for name in training_names}
            training_labels = {name: labels_by_run[name] for name in training_names}
            held_out_counts = {
                label: sum(labels_by_run[name] == label for name in held_out_names)
                for label in (NORMAL_ID, *FAULT_IDS)
            }
            if any(value == 0 for value in held_out_counts.values()) or len(
                set(held_out_counts.values())
            ) != 1:
                raise ValueError(
                    f"held-out fold {held_out_fold} is not class-balanced: "
                    f"{held_out_counts}"
                )
            model, convergence = fit_directional_glr_cuda(
                training_features,
                training_labels,
                ridge_lambda=ridge_lambda,
                **common,
            )
            fits.append(
                DirectionalGLRFoldFit(
                    ridge_lambda=ridge_lambda,
                    held_out_fold=held_out_fold,
                    training_run_names=training_names,
                    held_out_run_names=held_out_names,
                    model=model,
                    convergence=convergence,
                )
            )
    return tuple(fits)


def _score_tensor_batch(
    torch: Any,
    values: Any,
    model_tensors: dict[str, Any],
    *,
    valid_start: int,
    threshold: float | None,
) -> dict[str, Any]:
    batch, length, _ = values.shape
    eligible_values = values[:, valid_start:]
    logits_valid = eligible_values @ model_tensors["weights"].T
    logits_valid = logits_valid + model_tensors["bias"]
    z_valid = (
        logits_valid - model_tensors["normal_center"]
    ) / model_tensors["normal_scale"]
    clipped_valid = torch.clamp(z_valid, min=-Z_CLIP, max=Z_CLIP)
    increments = clipped_valid - model_tensors["kappa"]
    partial = torch.cumsum(increments, dim=1)
    augmented = torch.cat(
        (
            torch.zeros(
                (batch, 1, len(FAULT_IDS)),
                dtype=torch.float64,
                device=values.device,
            ),
            partial,
        ),
        dim=1,
    )
    running_minimum = torch.cummin(augmented, dim=1).values[:, 1:]
    cusum_valid = torch.clamp_min(partial - running_minimum, 0.0)

    path_shape = (batch, length, len(FAULT_IDS))
    logits = torch.full(
        path_shape,
        torch.nan,
        dtype=torch.float64,
        device=values.device,
    )
    standardized = torch.full_like(logits, torch.nan)
    clipped = torch.full_like(logits, torch.nan)
    cusum = torch.full_like(logits, torch.nan)
    logits[:, valid_start:] = logits_valid
    standardized[:, valid_start:] = z_valid
    clipped[:, valid_start:] = clipped_valid
    cusum[:, valid_start:] = cusum_valid

    first = valid_start + PERSISTENCE_WINDOW - 1
    rolling = cusum_valid.unfold(1, PERSISTENCE_WINDOW, 1)
    persistence_valid = torch.min(rolling, dim=-1).values
    component_persistence = torch.full(
        path_shape,
        -torch.inf,
        dtype=torch.float64,
        device=values.device,
    )
    component_persistence[:, first:] = persistence_valid
    envelope = torch.full(
        (batch, length),
        -torch.inf,
        dtype=torch.float64,
        device=values.device,
    )
    envelope[:, first:] = torch.max(persistence_valid, dim=2).values
    eligible = torch.zeros((batch, length), dtype=torch.bool, device=values.device)
    eligible[:, first:] = True
    alarm = torch.zeros_like(eligible)
    if threshold is not None:
        alarm = eligible & (envelope > threshold)
    return {
        "logits": logits,
        "standardized_logits": standardized,
        "clipped_evidence": clipped,
        "cusum": cusum,
        "component_persistence": component_persistence,
        "envelope": envelope,
        "eligible": eligible,
        "alarm": alarm,
    }


def score_directional_glr_cuda(
    model: DirectionalGLRModel,
    features_by_run: Mapping[str, np.ndarray],
    *,
    valid_start: int,
    device: str = "cuda",
    batch_size: int = 32,
) -> DirectionalGLRGPUResult:
    """Batch independent runs on CUDA; Page state is reset for every run."""
    if not isinstance(model, DirectionalGLRModel):
        raise TypeError("model must be a DirectionalGLRModel")
    if isinstance(valid_start, (bool, np.bool_)) or not isinstance(
        valid_start,
        (int, np.integer),
    ):
        raise ValueError("valid_start must be a nonnegative integer")
    valid_start = int(valid_start)
    if valid_start < 0:
        raise ValueError("valid_start must be a nonnegative integer")
    batch_size = _positive_integer(batch_size, name="batch_size")
    records = validate_feature_mapping(
        features_by_run,
        feature_count=model.feature_count,
        minimum_length=valid_start + PERSISTENCE_WINDOW,
        finite_start=valid_start,
        require_equal_length=True,
        label="directional CUDA scoring features",
    )
    sequence_length = int(records[0][1].shape[0])
    torch, target, runtime = _torch_and_device(device)
    model_tensors = {
        "weights": torch.tensor(model.weights, dtype=torch.float64, device=target),
        "bias": torch.tensor(model.bias, dtype=torch.float64, device=target),
        "normal_center": torch.tensor(
            model.normal_logit_center,
            dtype=torch.float64,
            device=target,
        ),
        "normal_scale": torch.tensor(
            model.normal_logit_scale,
            dtype=torch.float64,
            device=target,
        ),
        "kappa": torch.tensor(model.kappa, dtype=torch.float64, device=target),
    }
    by_run: dict[str, DirectionalGLRPath] = {}
    elapsed_ms = 0.0
    kernel_calls = 0
    torch.cuda.reset_peak_memory_stats(target)
    with _tf32_disabled(torch):
        with torch.inference_mode():
            for start in range(0, len(records), batch_size):
                chunk = records[start : start + batch_size]
                values = torch.as_tensor(
                    np.stack([matrix for _, matrix in chunk], axis=0),
                    dtype=torch.float64,
                    device=target,
                )
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                tensors = _score_tensor_batch(
                    torch,
                    values,
                    model_tensors,
                    valid_start=valid_start,
                    threshold=model.threshold,
                )
                end.record()
                end.synchronize()
                elapsed_ms += float(begin.elapsed_time(end))
                kernel_calls += 1
                arrays = {
                    name: tensor.detach().cpu().numpy()
                    for name, tensor in tensors.items()
                }
                for position, (name, _) in enumerate(chunk):
                    by_run[name] = DirectionalGLRPath(
                        fault_ids=FAULT_IDS,
                        logits=np.array(arrays["logits"][position], copy=True),
                        standardized_logits=np.array(
                            arrays["standardized_logits"][position],
                            copy=True,
                        ),
                        clipped_evidence=np.array(
                            arrays["clipped_evidence"][position],
                            copy=True,
                        ),
                        cusum=np.array(arrays["cusum"][position], copy=True),
                        component_persistence=np.array(
                            arrays["component_persistence"][position],
                            copy=True,
                        ),
                        envelope=np.array(arrays["envelope"][position], copy=True),
                        eligible=np.array(
                            arrays["eligible"][position],
                            dtype=bool,
                            copy=True,
                        ),
                        alarm=np.array(
                            arrays["alarm"][position],
                            dtype=bool,
                            copy=True,
                        ),
                        valid_start=valid_start,
                        threshold=model.threshold,
                    )
                del values, tensors
    if len(by_run) != len(records):
        raise RuntimeError("CUDA scorer did not return every named run")
    execution = {
        "backend": "pytorch_cuda_directional_glr_page_cusum",
        "device": str(target),
        "device_name": runtime["device_name"],
        "dtype": "float64",
        "tf32": False,
        "kernel_calls": kernel_calls,
        "sequences": len(records),
        "sequence_length": sequence_length,
        "time_steps": len(records) * sequence_length,
        "feature_count": model.feature_count,
        "fault_head_count": len(FAULT_IDS),
        "valid_start_zero_based": valid_start,
        "first_alarm_eligible_zero_based": (
            valid_start + PERSISTENCE_WINDOW - 1
        ),
        "z_clip": [-Z_CLIP, Z_CLIP],
        "page_state_reset_per_run": True,
        "persistence": "3_of_3_per_head_then_family_maximum",
        "cuda_event_elapsed_seconds": elapsed_ms / 1000.0,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(target)),
        "fallback_sequences": 0,
    }
    return DirectionalGLRGPUResult(
        by_run=by_run,
        execution=execution,
        runtime_environment=runtime,
        sequence_length=sequence_length,
        feature_count=model.feature_count,
    )


def validate_directional_glr_cuda(
    model: DirectionalGLRModel,
    representative_features: np.ndarray,
    *,
    valid_start: int,
    device: str = "cuda",
    atol: float = 1e-10,
) -> dict[str, object]:
    """Prove CPU/CUDA, duplicate, and batch-partition invariance."""
    reference = score_directional_glr_numpy(
        model,
        representative_features,
        valid_start=valid_start,
    )
    single = score_directional_glr_cuda(
        model,
        {"single": representative_features},
        valid_start=valid_start,
        device=device,
        batch_size=1,
    )
    duplicated_batch = score_directional_glr_cuda(
        model,
        {
            "duplicate_a": representative_features,
            "duplicate_b": representative_features,
        },
        valid_start=valid_start,
        device=device,
        batch_size=2,
    )
    duplicated_partitioned = score_directional_glr_cuda(
        model,
        {
            "duplicate_a": representative_features,
            "duplicate_b": representative_features,
        },
        valid_start=valid_start,
        device=device,
        batch_size=1,
    )
    cpu_cuda = compare_directional_paths(
        reference,
        single.by_run["single"],
        atol=atol,
    )
    duplicate_within_batch = compare_directional_paths(
        duplicated_batch.by_run["duplicate_a"],
        duplicated_batch.by_run["duplicate_b"],
        atol=atol,
    )
    batch_partition_a = compare_directional_paths(
        duplicated_batch.by_run["duplicate_a"],
        duplicated_partitioned.by_run["duplicate_a"],
        atol=atol,
    )
    batch_partition_b = compare_directional_paths(
        duplicated_batch.by_run["duplicate_b"],
        duplicated_partitioned.by_run["duplicate_b"],
        atol=atol,
    )
    return {
        "atol": float(atol),
        "cpu_cuda": cpu_cuda,
        "duplicate_within_batch": duplicate_within_batch,
        "batch_partition_a": batch_partition_a,
        "batch_partition_b": batch_partition_b,
        "cpu_cuda_alarm_exact": bool(cpu_cuda["alarm_exact"]),
        "duplicate_alarm_exact": bool(duplicate_within_batch["alarm_exact"]),
        "batch_partition_alarm_exact": bool(
            batch_partition_a["alarm_exact"] and batch_partition_b["alarm_exact"]
        ),
        "runtime_environment": single.runtime_environment,
        "single_execution": single.execution,
        "duplicate_batch_execution": duplicated_batch.execution,
        "partitioned_execution": duplicated_partitioned.execution,
    }


__all__ = [
    "DirectionalGLRFoldFit",
    "DirectionalGLRGPUResult",
    "fit_directional_glr_cuda",
    "fit_directional_glr_grid_cuda",
    "score_directional_glr_cuda",
    "validate_directional_glr_cuda",
]
