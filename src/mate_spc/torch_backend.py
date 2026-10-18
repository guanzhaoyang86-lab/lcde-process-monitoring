"""Optional CUDA kernels for compact batched posterior scoring.

The statistical model is still fitted by the NumPy implementation.  This
module transfers the frozen parameters to CUDA and evaluates several equally
sized, independent sequences in parallel.  PyTorch is imported lazily so the
core package keeps working when the optional dependency is absent.
"""

from __future__ import annotations

from contextlib import contextmanager
import importlib
from typing import Any

import numpy as np


@contextmanager
def _tf32_disabled(torch: Any) -> Any:
    """Temporarily disable TF32 without mutating the caller's global setting."""
    matmul_backend = torch.backends.cuda.matmul
    cudnn_backend = getattr(torch.backends, "cudnn", None)
    previous_matmul = bool(matmul_backend.allow_tf32)
    previous_cudnn = (
        bool(cudnn_backend.allow_tf32) if cudnn_backend is not None else None
    )
    matmul_backend.allow_tf32 = False
    if cudnn_backend is not None:
        cudnn_backend.allow_tf32 = False
    try:
        yield
    finally:
        matmul_backend.allow_tf32 = previous_matmul
        if cudnn_backend is not None and previous_cudnn is not None:
            cudnn_backend.allow_tf32 = previous_cudnn


def _load_cuda_torch(device: str, dtype: str) -> tuple[Any, Any, Any]:
    """Return torch, a validated CUDA device, and the requested dtype."""
    try:
        torch = importlib.import_module("torch")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "PyTorch with CUDA support is required for the torch backend"
        ) from exc

    try:
        target = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"invalid torch device {device!r}") from exc
    if target.type != "cuda":
        raise ValueError("score_incomplete_batch requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this PyTorch process; activate a "
            "CUDA-enabled environment before using the torch backend"
        )
    try:
        torch.cuda.get_device_properties(target)
    except (AssertionError, RuntimeError) as exc:
        raise RuntimeError(f"CUDA device {device!r} is not accessible") from exc

    normalized_dtype = str(dtype).lower().replace("torch.", "")
    if normalized_dtype not in {"float64", "double"}:
        raise ValueError(
            "the compact CUDA backend currently requires dtype='float64' "
            "to preserve the NumPy reference semantics"
        )
    return torch, target, torch.float64


def _as_square_parameter(value: object, name: str, dimension: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (dimension, dimension) or not np.isfinite(array).all():
        raise ValueError(f"monitor parameter {name} has an invalid shape or value")
    return array


def _validate_inputs(
    monitor: object,
    x_batch: np.ndarray,
    mask_batch: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, object, int]:
    if not bool(getattr(monitor, "fitted_", False)):
        raise RuntimeError("PosteriorIntegratedMonitor has not been fitted")
    filter_model = getattr(monitor, "filter_model", None)
    if filter_model is None or not bool(getattr(filter_model, "fitted_", False)):
        raise RuntimeError("the monitor's LinearGaussianFilter has not been fitted")
    if int(getattr(filter_model, "n_exogenous_", 0)) != 0:
        raise NotImplementedError(
            "the compact CUDA backend currently supports monitors without "
            "exogenous variables"
        )

    values = np.asarray(x_batch, dtype=np.float64)
    observed = np.asarray(mask_batch, dtype=bool)
    if values.ndim != 3:
        raise ValueError("x_batch must have shape (batch, time, feature)")
    if observed.shape != values.shape:
        raise ValueError("mask_batch must have the same shape as x_batch")
    if values.shape[0] < 1 or values.shape[1] < 2 or values.shape[2] < 1:
        raise ValueError("x_batch must contain at least one two-sample sequence")
    if np.isinf(values).any():
        raise ValueError("x_batch contains infinite values")
    if np.any(observed & ~np.isfinite(values)):
        raise ValueError("entries marked as observed must be finite")

    dimension = int(getattr(filter_model, "dimension_", -1))
    if values.shape[2] != dimension:
        raise ValueError(
            "x_batch feature count differs from the fitted monitor dimension"
        )
    return values, observed, filter_model, dimension


def _symmetrize_batch(matrix: Any) -> Any:
    """Remove round-off asymmetry without an expensive eigendecomposition."""
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def _quadratic_batch(torch: Any, mean: Any, matrix: Any) -> Any:
    values = torch.einsum("bi,ij,bj->b", mean, matrix, mean)
    return torch.clamp(values, min=0.0)


def _trace_product_batch(torch: Any, covariance: Any, matrix: Any) -> Any:
    values = torch.einsum("ij,bji->b", matrix, covariance)
    return torch.clamp(values, min=0.0)


def _run_compact_filter(
    torch: Any,
    x_tensor: Any,
    mask_tensor: Any,
    parameters: dict[str, Any],
    solver: str,
) -> tuple[dict[str, Any], Any, Any]:
    """Run a compact batch with a fast solve or a robust replay solve."""
    if solver not in {"cholesky", "pinv"}:
        raise ValueError(f"unknown CUDA solve mode {solver!r}")

    batch_size, sequence_length, dimension = x_tensor.shape
    torch_dtype = x_tensor.dtype
    target = x_tensor.device
    location = parameters["location"]
    scale = parameters["scale"]
    intercept = parameters["intercept"]
    transition = parameters["transition"]
    innovation_covariance = parameters["innovation_covariance"]
    initial_covariance = parameters["initial_covariance"]
    baseline_intercept = parameters["baseline_intercept"]
    t2_matrix = parameters["t2_matrix"]
    spe_matrix = parameters["spe_matrix"]
    combined_matrix = parameters["combined_matrix"]

    standardized_observed = (x_tensor - location) / scale
    means = torch.empty(
        (batch_size, sequence_length, dimension),
        dtype=torch_dtype,
        device=target,
    )
    normalized_innovation = torch.zeros(
        (batch_size, sequence_length), dtype=torch_dtype, device=target
    )
    outputs = {
        "mean_z": means,
        "normalized_innovation": normalized_innovation,
        "t2_location": torch.empty_like(normalized_innovation),
        "spe_location": torch.empty_like(normalized_innovation),
        "t2_uncertainty": torch.empty_like(normalized_innovation),
        "spe_uncertainty": torch.empty_like(normalized_innovation),
        "location_energy": torch.empty_like(normalized_innovation),
        "uncertainty_energy": torch.empty_like(normalized_innovation),
        "posterior_trace": torch.empty_like(normalized_innovation),
    }

    previous_mean = torch.zeros(
        (batch_size, dimension), dtype=torch_dtype, device=target
    )
    previous_covariance = initial_covariance.expand(batch_size, -1, -1).clone()
    identity = torch.eye(dimension, dtype=torch_dtype, device=target)
    failed_sequences = torch.zeros(batch_size, dtype=torch.bool, device=target)
    factorization_failures = torch.zeros((), dtype=torch.int64, device=target)

    for time_index in range(sequence_length):
        if time_index == 0:
            prior_mean = intercept.expand(batch_size, -1)
            prior_covariance = initial_covariance.expand(
                batch_size, -1, -1
            ).clone()
        else:
            prior_mean = intercept + previous_mean @ transition.transpose(0, 1)
            prior_covariance = (
                innovation_covariance
                + transition.unsqueeze(0)
                @ previous_covariance
                @ transition.transpose(0, 1).unsqueeze(0)
            )
            # Q + A P A' is PSD in exact arithmetic.  Symmetrization removes
            # round-off asymmetry without an eigendecomposition at every step.
            prior_covariance = _symmetrize_batch(prior_covariance)

        current_mask = mask_tensor[:, time_index]
        mask_float = current_mask.to(dtype=torch_dtype)
        missing_float = 1.0 - mask_float
        observed_outer = mask_float.unsqueeze(-1) * mask_float.unsqueeze(-2)
        missing_outer = missing_float.unsqueeze(-1) * missing_float.unsqueeze(-2)

        observed_count = mask_float.sum(dim=1)
        observed_trace = (
            prior_covariance.diagonal(dim1=-2, dim2=-1) * mask_float
        ).sum(dim=1)
        jitter = torch.clamp(
            observed_trace / torch.clamp(observed_count, min=1.0), min=1.0
        ) * 1e-10
        system = prior_covariance * observed_outer
        system = system + torch.diag_embed(
            missing_float + jitter.unsqueeze(1) * mask_float
        )

        innovation = (
            standardized_observed[:, time_index] - prior_mean
        ) * mask_float
        right_hand_side = torch.cat(
            [
                innovation.unsqueeze(-1),
                mask_float.unsqueeze(-1) * prior_covariance,
            ],
            dim=-1,
        )
        if solver == "cholesky":
            factor, info = torch.linalg.cholesky_ex(system, check_errors=False)
            failed = info != 0
            failed_sequences |= failed
            factorization_failures += failed.sum(dtype=torch.int64)
            # Failed sequences are replayed from t=0 below.  Using identity for
            # their provisional update keeps the rest of this batch finite.
            safe_factor = torch.where(
                failed.reshape(-1, 1, 1),
                identity.expand(batch_size, -1, -1),
                factor,
            )
            solution = torch.cholesky_solve(right_hand_side, safe_factor)
        else:
            inverse = torch.linalg.pinv(system, hermitian=True, rtol=1e-10)
            solution = inverse @ right_hand_side

        prior_observed_columns = prior_covariance * mask_float.unsqueeze(-2)
        update = prior_observed_columns @ solution
        posterior_mean = prior_mean + update[:, :, 0]
        posterior_mean = torch.where(
            current_mask,
            standardized_observed[:, time_index],
            posterior_mean,
        )
        posterior_covariance_raw = (
            prior_covariance - update[:, :, 1:]
        ) * missing_outer
        conditional_covariance = _symmetrize_batch(
            posterior_covariance_raw
        ) * missing_outer
        has_observation = current_mask.any(dim=1).reshape(-1, 1, 1)
        has_missing = (~current_mask).any(dim=1).reshape(-1, 1, 1)
        posterior_covariance = torch.where(
            has_observation & has_missing,
            conditional_covariance,
            torch.where(
                has_observation,
                torch.zeros_like(prior_covariance),
                prior_covariance,
            ),
        )

        solved_innovation = solution[:, :, 0]
        normalized_innovation[:, time_index] = torch.where(
            observed_count > 0.0,
            (innovation * solved_innovation).sum(dim=1)
            / torch.clamp(observed_count, min=1.0),
            torch.zeros_like(observed_count),
        )
        means[:, time_index] = posterior_mean

        monitoring_mean = posterior_mean - baseline_intercept
        outputs["t2_location"][:, time_index] = _quadratic_batch(
            torch, monitoring_mean, t2_matrix
        )
        outputs["spe_location"][:, time_index] = _quadratic_batch(
            torch, monitoring_mean, spe_matrix
        )
        outputs["t2_uncertainty"][:, time_index] = _trace_product_batch(
            torch, posterior_covariance, t2_matrix
        )
        outputs["spe_uncertainty"][:, time_index] = _trace_product_batch(
            torch, posterior_covariance, spe_matrix
        )
        outputs["location_energy"][:, time_index] = _quadratic_batch(
            torch, monitoring_mean, combined_matrix
        )
        outputs["uncertainty_energy"][:, time_index] = _trace_product_batch(
            torch, posterior_covariance, combined_matrix
        )
        outputs["posterior_trace"][:, time_index] = (
            posterior_covariance.diagonal(dim1=-2, dim2=-1).sum(dim=1)
            / dimension
        )
        previous_mean = posterior_mean
        previous_covariance = posterior_covariance

    reconstructed = means * scale + location
    outputs["reconstructed"] = torch.where(mask_tensor, x_tensor, reconstructed)
    return outputs, failed_sequences, factorization_failures


def score_incomplete_batch(
    monitor: object,
    x_batch: np.ndarray,
    mask_batch: np.ndarray,
    device: str = "cuda",
    dtype: str = "float64",
) -> dict[str, object]:
    """Score independent incomplete sequences with a fitted monitor on CUDA.

    Parameters
    ----------
    monitor:
        A fitted :class:`PosteriorIntegratedMonitor`.  Its frozen NumPy
        parameters are used verbatim; this function does not refit the model.
    x_batch, mask_batch:
        Arrays with shape ``(batch, time, feature)``.  All sequences must have
        the same length.  Missing values may be NaN and must be marked false in
        ``mask_batch``.
    device:
        CUDA device accepted by ``torch.device``, for example ``"cuda"`` or
        ``"cuda:0"``.
    dtype:
        Currently restricted to ``"float64"`` for comparison with the NumPy
        reference implementation.

    Returns
    -------
    dict
        NumPy arrays for posterior means, reconstruction, normalized
        innovation, location/uncertainty energy components, and posterior
        trace.  The time-varying posterior covariance is intentionally not
        retained.  ``elapsed_ms`` is measured with CUDA events and covers
        host-to-device copies plus CUDA computation, but not result copies back
        to host memory.

    Notes
    -----
    The time recursion remains sequential.  Parallelism is across independent
    sequences.  A fixed-dimensional masked Gaussian conditioning identity is
    used so different observed subsets can share one batched factorization.
    """
    values, observed, filter_model, dimension = _validate_inputs(
        monitor, x_batch, mask_batch
    )
    torch, target, torch_dtype = _load_cuda_torch(device, dtype)

    location_np = np.asarray(filter_model.location_, dtype=np.float64)
    scale_np = np.asarray(filter_model.scale_, dtype=np.float64)
    if (
        location_np.shape != (dimension,)
        or scale_np.shape != (dimension,)
        or not np.isfinite(location_np).all()
        or not np.isfinite(scale_np).all()
        or np.any(scale_np <= 0.0)
    ):
        raise ValueError("the fitted filter has invalid location or scale parameters")

    intercept_np = np.asarray(filter_model.intercept_, dtype=np.float64)
    baseline_np = np.asarray(monitor.baseline_intercept_, dtype=np.float64)
    if (
        intercept_np.shape != (dimension,)
        or baseline_np.shape != (dimension,)
        or not np.isfinite(intercept_np).all()
        or not np.isfinite(baseline_np).all()
    ):
        raise ValueError("the fitted monitor has invalid intercept parameters")

    transition_np = _as_square_parameter(
        filter_model.transition_, "transition_", dimension
    )
    innovation_covariance_np = _as_square_parameter(
        filter_model.innovation_covariance_,
        "innovation_covariance_",
        dimension,
    )
    initial_covariance_np = _as_square_parameter(
        filter_model.initial_covariance_, "initial_covariance_", dimension
    )
    t2_matrix_np = _as_square_parameter(
        monitor.t2_matrix_, "t2_matrix_", dimension
    )
    spe_matrix_np = _as_square_parameter(
        monitor.spe_matrix_, "spe_matrix_", dimension
    )
    combined_matrix_np = _as_square_parameter(
        monitor.combined_matrix_, "combined_matrix_", dimension
    )

    batch_size, sequence_length, _ = values.shape
    safe_values = np.where(observed, values, location_np)

    with torch.cuda.device(target), torch.inference_mode(), _tf32_disabled(torch):
        # The calculation is double precision, but disabling TF32 explicitly
        # protects this routine if parameters are generalized in the future.
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        x_tensor = torch.as_tensor(safe_values, dtype=torch_dtype, device=target)
        mask_tensor = torch.as_tensor(observed, dtype=torch.bool, device=target)
        location = torch.as_tensor(location_np, dtype=torch_dtype, device=target)
        scale = torch.as_tensor(scale_np, dtype=torch_dtype, device=target)
        intercept = torch.as_tensor(intercept_np, dtype=torch_dtype, device=target)
        transition = torch.as_tensor(
            transition_np, dtype=torch_dtype, device=target
        )
        innovation_covariance = torch.as_tensor(
            innovation_covariance_np, dtype=torch_dtype, device=target
        )
        initial_covariance = torch.as_tensor(
            initial_covariance_np, dtype=torch_dtype, device=target
        )
        baseline_intercept = torch.as_tensor(
            baseline_np, dtype=torch_dtype, device=target
        )
        t2_matrix = torch.as_tensor(t2_matrix_np, dtype=torch_dtype, device=target)
        spe_matrix = torch.as_tensor(
            spe_matrix_np, dtype=torch_dtype, device=target
        )
        combined_matrix = torch.as_tensor(
            combined_matrix_np, dtype=torch_dtype, device=target
        )

        parameters = {
            "location": location,
            "scale": scale,
            "intercept": intercept,
            "transition": transition,
            "innovation_covariance": innovation_covariance,
            "initial_covariance": initial_covariance,
            "baseline_intercept": baseline_intercept,
            "t2_matrix": t2_matrix,
            "spe_matrix": spe_matrix,
            "combined_matrix": combined_matrix,
        }
        outputs, failed_sequences, failure_count = _run_compact_filter(
            torch,
            x_tensor,
            mask_tensor,
            parameters,
            solver="cholesky",
        )
        # This is the only failure-status synchronization in the batch.  If a
        # sequence failed at any time step, replay it from t=0 because its
        # later posterior state depends on the failed update.
        failed_indices = torch.nonzero(
            failed_sequences, as_tuple=False
        ).flatten()
        fallback_sequence_count = int(failed_indices.numel())
        factorization_failure_count = int(failure_count.item())
        if fallback_sequence_count:
            fallback_outputs, _, _ = _run_compact_filter(
                torch,
                x_tensor.index_select(0, failed_indices),
                mask_tensor.index_select(0, failed_indices),
                parameters,
                solver="pinv",
            )
            for key, fallback_value in fallback_outputs.items():
                outputs[key].index_copy_(0, failed_indices, fallback_value)

        end_event.record()
        end_event.synchronize()
        elapsed_ms = float(start_event.elapsed_time(end_event))

        means = outputs["mean_z"]
        reconstructed = outputs["reconstructed"]
        normalized_innovation = outputs["normalized_innovation"]
        t2_location = outputs["t2_location"]
        spe_location = outputs["spe_location"]
        t2_uncertainty = outputs["t2_uncertainty"]
        spe_uncertainty = outputs["spe_uncertainty"]
        location_energy = outputs["location_energy"]
        uncertainty_energy = outputs["uncertainty_energy"]
        posterior_trace = outputs["posterior_trace"]

        def to_numpy(tensor: Any) -> np.ndarray:
            return tensor.detach().cpu().numpy()

        result: dict[str, object] = {
            "mean_z": to_numpy(means),
            "reconstructed": to_numpy(reconstructed),
            "normalized_innovation": to_numpy(normalized_innovation),
            "nis": to_numpy(normalized_innovation),
            "observed_mask": observed.copy(),
            "t2_location": to_numpy(t2_location),
            "spe_location": to_numpy(spe_location),
            "t2_uncertainty": to_numpy(t2_uncertainty),
            "spe_uncertainty": to_numpy(spe_uncertainty),
            "location_energy": to_numpy(location_energy),
            "uncertainty_energy": to_numpy(uncertainty_energy),
            "posterior_trace": to_numpy(posterior_trace),
            "elapsed_ms": elapsed_ms,
            "device": str(target),
            "device_name": str(torch.cuda.get_device_name(target)),
            "dtype": "float64",
            "batch_size": int(batch_size),
            "sequence_length": int(sequence_length),
            "dimension": int(dimension),
            "factorization_failure_count": factorization_failure_count,
            "fallback_sequence_count": fallback_sequence_count,
            "timing": {
                "kind": "cuda_event",
                "elapsed_ms": elapsed_ms,
                "includes_host_to_device": True,
                "includes_device_to_host": False,
            },
        }
    return result


__all__ = ["score_incomplete_batch"]
