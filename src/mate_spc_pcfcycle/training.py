"""Deterministic tensor-only training for PCF-CycleRank-Scan v6."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
from torch import Tensor

from .loss import (
    FAULT_IDS,
    LossOutput,
    composite_supervised_loss,
    discriminator_lsgan_loss,
    generator_lsgan_cycle_loss,
)
from .model import (
    BUILD_SEED,
    EMBEDDING_DIM,
    FEATURE_DIM,
    MONITOR_TRAIN_ENDPOINTS,
    MATURE_FAULT_ENDPOINTS,
    NETWORK_DTYPE,
    PCFCycleModel,
    PREFIX_ENDPOINTS,
    RAW_CHANNELS,
    RAW_LENGTH,
    RESIDUAL_SIGNATURE_DIM,
    TRAIN_ENDPOINTS,
    EndpointScores,
    Stage2Detector,
    build_pcfcycle_model,
    load_pcfcycle_checkpoint,
    residual_signature,
    save_pcfcycle_checkpoint,
    score_endpoint_pairs,
)


CANDIDATE_ID = "pcf-cyclerank-scan-v6-fixed"
TRAINING_SCHEMA = "mate_spc_pcfcycle_training_v6"
CLASS_IDS = (0, *FAULT_IDS)
ALLOWED_PRODUCTION_RUNS = (60, 100)
STAGE1_EPOCHS = 120
STAGE1_BATCH_SIZE = 128
STAGE1_LEARNING_RATE = 1.0e-4
STAGE1_WEIGHT_DECAY = 1.0e-4
STAGE1_BETAS = (0.9, 0.999)
STAGE1_EPSILON = 1.0e-8
STAGE2_EPOCHS = 80
STAGE2_RUNS_PER_CLASS_BATCH = 2
STAGE2_LEARNING_RATE = 3.0e-4
STAGE2_WEIGHT_DECAY = 1.0e-4
STAGE2_BETAS = (0.9, 0.999)
STAGE2_EPSILON = 1.0e-8
PRECOMPUTE_BATCH_SIZE = 256
BFEAT_MAD_CONSISTENCY = 1.4826
BFEAT_SCALE_FLOOR = 1.0e-6
BFEAT_CLIP_BOUND = 10.0

PREFIX_COUNT = len(PREFIX_ENDPOINTS)
MONITOR_COUNT = len(MONITOR_TRAIN_ENDPOINTS)
MATURE_COUNT = len(MATURE_FAULT_ENDPOINTS)
TRAIN_ENDPOINT_COUNT = len(TRAIN_ENDPOINTS)
PREFIX_SLICE = slice(0, PREFIX_COUNT)
MONITOR_SLICE = slice(PREFIX_COUNT, TRAIN_ENDPOINT_COUNT)
MATURE_START_INDEX = TRAIN_ENDPOINTS.index(MATURE_FAULT_ENDPOINTS[0])
MATURE_SLICE = slice(MATURE_START_INDEX, TRAIN_ENDPOINT_COUNT)
IGNORED_FAULT_SLICE = slice(PREFIX_COUNT, MATURE_START_INDEX)


if (
    PREFIX_COUNT != 8
    or MONITOR_COUNT != 85
    or MATURE_COUNT != 77
    or TRAIN_ENDPOINT_COUNT != 93
    or MATURE_START_INDEX != 16
):  # pragma: no cover
    raise RuntimeError("PCF-CycleRank endpoint contract drifted")


def bfeat_normalization_contract(fit_runs: int) -> dict[str, Any]:
    """Describe the external fit-only transform expected by the neural core."""

    if isinstance(fit_runs, bool) or not isinstance(fit_runs, int):
        raise ValueError("fit_runs must be an integer")
    if fit_runs not in ALLOWED_PRODUCTION_RUNS:
        raise ValueError("fit_runs must be one of 60 or 100")
    return {
        "fit_population": f"{fit_runs} fit-healthy runs x 85 monitor endpoints",
        "fit_runs": fit_runs,
        "fit_rows": fit_runs * MONITOR_COUNT,
        "center": "coordinatewise_median",
        "scale": "1.4826 * coordinatewise_median_absolute_deviation",
        "mad_consistency": BFEAT_MAD_CONSISTENCY,
        "scale_floor": BFEAT_SCALE_FLOOR,
        "clip": [-BFEAT_CLIP_BOUND, BFEAT_CLIP_BOUND],
        "model_input": "already_standardized_float32",
    }


BFEAT_NORMALIZATION = bfeat_normalization_contract(60)


@dataclass(frozen=True)
class TrainingHistory:
    """Serializable history for one frozen training stage."""

    stage: str
    seed: int
    configuration: Mapping[str, Any]
    epochs: tuple[Mapping[str, float], ...]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "seed": self.seed,
            "configuration": dict(self.configuration),
            "epochs": [dict(row) for row in self.epochs],
        }

    @classmethod
    def from_metadata(cls, value: Any, *, expected_stage: str) -> "TrainingHistory":
        if not isinstance(value, Mapping) or value.get("stage") != expected_stage:
            raise ValueError(f"checkpoint {expected_stage} history is invalid")
        if set(value) != {"stage", "seed", "configuration", "epochs"}:
            raise ValueError(f"checkpoint {expected_stage} history fields changed")
        seed = value.get("seed")
        configuration = value.get("configuration")
        rows = value.get("epochs")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"checkpoint {expected_stage} seed is invalid")
        if not isinstance(configuration, Mapping) or not isinstance(rows, (list, tuple)):
            raise ValueError(f"checkpoint {expected_stage} history is invalid")
        converted: list[dict[str, float]] = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"checkpoint {expected_stage} epoch row is invalid")
            numeric: dict[str, float] = {}
            for key, item in row.items():
                if isinstance(item, bool) or not isinstance(item, (int, float)):
                    raise ValueError(
                        f"checkpoint {expected_stage} epoch metric is invalid"
                    )
                number = float(item)
                if not math.isfinite(number):
                    raise ValueError(
                        f"checkpoint {expected_stage} epoch metric is non-finite"
                    )
                numeric[str(key)] = number
            converted.append(numeric)
        return cls(expected_stage, seed, dict(configuration), tuple(converted))


@dataclass(frozen=True)
class Stage2LossInputs:
    """Exact tensor populations passed to the four-part supervised loss."""

    detection_logits: Tensor
    binary_targets: Tensor
    post_logits: Tensor
    prefix_logits: Tensor
    embeddings: Tensor
    contrastive_class_labels: Tensor
    attribution_logits: Tensor
    attribution_fault_ids: Tensor

    def loss(self) -> LossOutput:
        return composite_supervised_loss(
            detection_logits=self.detection_logits,
            binary_targets=self.binary_targets,
            post_logits=self.post_logits,
            prefix_logits=self.prefix_logits,
            embeddings=self.embeddings,
            contrastive_class_labels=self.contrastive_class_labels,
            attribution_logits=self.attribution_logits,
            attribution_fault_ids=self.attribution_fault_ids,
        )


def _history_integer(configuration: Mapping[str, Any], key: str, *, stage: str) -> int:
    value = configuration.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{stage} configuration {key} must be an integer")
    return value


def _validate_epoch_rows(
    history: TrainingHistory,
    *,
    expected_epochs: int,
    expected_batches: int,
    expected_fields: set[str],
) -> None:
    if len(history.epochs) != expected_epochs:
        raise ValueError(
            f"{history.stage} history must contain exactly {expected_epochs} epochs"
        )
    for index, row in enumerate(history.epochs, start=1):
        if not isinstance(row, Mapping) or set(row) != expected_fields:
            raise ValueError(f"{history.stage} epoch fields changed")
        for value in row.values():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{history.stage} epoch metrics must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{history.stage} epoch metrics must be finite")
        if float(row["epoch"]) != float(index):
            raise ValueError(f"{history.stage} epoch numbering changed")
        if float(row["batches"]) != float(expected_batches):
            raise ValueError(f"{history.stage} batch count changed")


def _validate_training_histories(
    stage1: TrainingHistory, stage2: TrainingHistory
) -> tuple[int, bool, int]:
    if not isinstance(stage1, TrainingHistory) or not isinstance(stage2, TrainingHistory):
        raise TypeError("both training histories must be TrainingHistory instances")
    if stage1.stage != "stage1" or stage2.stage != "stage2":
        raise ValueError("training history stage labels changed")
    if (
        isinstance(stage1.seed, bool)
        or not isinstance(stage1.seed, int)
        or isinstance(stage2.seed, bool)
        or not isinstance(stage2.seed, int)
        or stage1.seed != stage2.seed
    ):
        raise ValueError("training histories must share one integer seed")
    if not isinstance(stage1.configuration, Mapping) or not isinstance(
        stage2.configuration, Mapping
    ):
        raise ValueError("training history configurations must be mappings")
    config1 = stage1.configuration
    config2 = stage2.configuration
    synthetic1 = config1.get("synthetic_test")
    synthetic2 = config2.get("synthetic_test")
    if not isinstance(synthetic1, bool) or not isinstance(synthetic2, bool):
        raise ValueError("training history synthetic flags must be boolean")
    if synthetic1 != synthetic2:
        raise ValueError("training histories disagree on synthetic status")
    synthetic_test = synthetic1
    runs1 = _history_integer(config1, "runs", stage="stage1")
    runs2 = _history_integer(config2, "runs_per_class", stage="stage2")
    if runs1 != runs2 or runs1 < 2 or runs1 % STAGE2_RUNS_PER_CLASS_BATCH:
        raise ValueError("training histories require the same positive even run count")
    production1 = _history_integer(
        config1, "production_runs_per_class", stage="stage1"
    )
    production2 = _history_integer(
        config2, "production_runs_per_class", stage="stage2"
    )
    if production1 != production2 or production1 not in ALLOWED_PRODUCTION_RUNS:
        raise ValueError("training histories have an invalid production run contract")
    if not synthetic_test and runs1 != production1:
        raise ValueError("production histories must contain exactly 60 or 100 runs")
    if not synthetic_test and stage1.seed != BUILD_SEED:
        raise ValueError(f"production histories must use frozen seed {BUILD_SEED}")

    stage1_epochs = _history_integer(config1, "epochs", stage="stage1")
    stage2_epochs = _history_integer(config2, "epochs", stage="stage2")
    monitor_count = _history_integer(
        config1, "monitor_endpoints_per_run", stage="stage1"
    )
    if synthetic_test:
        if stage1_epochs < 1 or stage2_epochs < 1 or monitor_count < 1:
            raise ValueError("synthetic history sizes must be positive")
    elif (
        stage1_epochs != STAGE1_EPOCHS
        or stage2_epochs != STAGE2_EPOCHS
        or monitor_count != MONITOR_COUNT
    ):
        raise ValueError("production history epoch/endpoint contract changed")
    fit_rows = runs1 * monitor_count
    stage1_batches = math.ceil(fit_rows / STAGE1_BATCH_SIZE)
    stage2_batches = runs1 // STAGE2_RUNS_PER_CLASS_BATCH
    expected1 = {
        "epochs": stage1_epochs,
        "batch_size": STAGE1_BATCH_SIZE,
        "learning_rate": STAGE1_LEARNING_RATE,
        "weight_decay": STAGE1_WEIGHT_DECAY,
        "betas": list(STAGE1_BETAS),
        "epsilon": STAGE1_EPSILON,
        "optimizer": "AdamW",
        "drop_last": False,
        "early_stopping": False,
        "runs": runs1,
        "monitor_endpoints_per_run": monitor_count,
        "fit_rows": fit_rows,
        "production_runs_per_class": production1,
        "synthetic_test": synthetic_test,
    }
    expected2 = {
        "epochs": stage2_epochs,
        "learning_rate": STAGE2_LEARNING_RATE,
        "weight_decay": STAGE2_WEIGHT_DECAY,
        "betas": list(STAGE2_BETAS),
        "epsilon": STAGE2_EPSILON,
        "optimizer": "AdamW",
        "drop_last": False,
        "early_stopping": False,
        "runs_per_class": runs2,
        "production_runs_per_class": production2,
        "runs_per_class_batch": STAGE2_RUNS_PER_CLASS_BATCH,
        "batches_per_epoch": stage2_batches,
        "prefix_endpoints": list(PREFIX_ENDPOINTS),
        "monitor_endpoints": list(MONITOR_TRAIN_ENDPOINTS),
        "mature_fault_endpoints": list(MATURE_FAULT_ENDPOINTS),
        "ignored_fault_endpoints": list(TRAIN_ENDPOINTS[IGNORED_FAULT_SLICE]),
        "synthetic_test": synthetic_test,
    }
    if dict(config1) != expected1:
        raise ValueError("stage1 optimizer/batch configuration changed")
    if dict(config2) != expected2:
        raise ValueError("stage2 optimizer/endpoint/batch configuration changed")
    if (
        config1["drop_last"] is not False
        or config1["early_stopping"] is not False
        or config2["drop_last"] is not False
        or config2["early_stopping"] is not False
    ):
        raise ValueError("training drop/early-stop flags must be boolean false")
    _validate_epoch_rows(
        stage1,
        expected_epochs=stage1_epochs,
        expected_batches=stage1_batches,
        expected_fields={"epoch", "discriminator_loss", "generator_loss", "batches"},
    )
    _validate_epoch_rows(
        stage2,
        expected_epochs=stage2_epochs,
        expected_batches=stage2_batches,
        expected_fields={
            "epoch",
            "total",
            "balanced_bce",
            "paired_rank",
            "supervised_contrastive",
            "mature_fault_attribution",
            "batches",
        },
    )
    return runs1, synthetic_test, stage1.seed


@dataclass(frozen=True)
class PCFCycleTrainingResult:
    """A trained model and both histories for an auditable checkpoint."""

    model: PCFCycleModel
    stage1_history: TrainingHistory
    stage2_history: TrainingHistory

    def _run_contract(self) -> tuple[int, bool, int]:
        return _validate_training_histories(
            self.stage1_history, self.stage2_history
        )

    def metadata(self, *, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        runs, synthetic_test, _ = self._run_contract()
        if synthetic_test:
            normalization = {
                "synthetic_test": True,
                "fit_runs": runs,
                "fit_rows": self.stage1_history.configuration.get("fit_rows"),
                "model_input": "synthetic_standardized_float32",
            }
        else:
            normalization = bfeat_normalization_contract(runs)
        document: dict[str, Any] = {
            "candidate_id": CANDIDATE_ID,
            "training_schema": TRAINING_SCHEMA,
            "runs_per_class": runs,
            "synthetic_test": synthetic_test,
            "bfeat_normalization": normalization,
            "stage1_history": self.stage1_history.to_metadata(),
            "stage2_history": self.stage2_history.to_metadata(),
        }
        if extra is not None:
            document["extra"] = dict(extra)
        return document

    def save(
        self,
        path: str | Path,
        *,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        _, _, seed = self._run_contract()
        return save_pcfcycle_checkpoint(
            self.model,
            path,
            metadata=self.metadata(extra=extra_metadata),
            build_seed=seed,
        )


def _require_model(value: Any) -> PCFCycleModel:
    if not isinstance(value, PCFCycleModel):
        raise TypeError("model must be a PCFCycleModel")
    return value


def _model_device(value: PCFCycleModel) -> torch.device:
    try:
        return next(value.parameters()).device
    except StopIteration as error:  # pragma: no cover
        raise RuntimeError("PCFCycleModel unexpectedly has no parameters") from error


def _require_float32(value: Any, *, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != NETWORK_DTYPE:
        raise ValueError(f"{name} must use torch.float32")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return value


def _require_clipped_bfeat(value: Tensor, *, name: str) -> None:
    if bool((value.abs() > BFEAT_CLIP_BOUND).any().item()):
        raise ValueError(
            f"{name} must be fit-only standardized and clipped to [-10,10]"
        )


def assert_finite_model_parameters(model: PCFCycleModel, *, stage: str) -> None:
    """Fail closed if any trained parameter is NaN or infinite."""

    value = _require_model(model)
    if not isinstance(stage, str) or not stage:
        raise ValueError("stage must be a nonempty string")
    nonfinite = [
        name
        for name, parameter in value.named_parameters()
        if not bool(torch.isfinite(parameter.detach()).all().item())
    ]
    if nonfinite:
        joined = ", ".join(nonfinite[:5])
        raise FloatingPointError(f"{stage} produced nonfinite parameters: {joined}")


def _require_positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_production_runs(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("runs_per_class must be an integer")
    if value not in ALLOWED_PRODUCTION_RUNS:
        raise ValueError("runs_per_class must be one of 60 or 100")
    return value


def _resolve_epochs(
    required: int,
    override: int | None,
    *,
    synthetic_test: bool,
    stage: str,
) -> int:
    if not isinstance(synthetic_test, bool):
        raise TypeError("synthetic_test must be boolean")
    if override is None:
        return required
    if not synthetic_test:
        raise ValueError(f"{stage} epoch override is allowed only for synthetic tests")
    return _require_positive_integer(override, name=f"{stage}_epochs")


def _validate_seed(seed: Any) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    return seed


@contextmanager
def _deterministic_training(seed: int, device: torch.device) -> Iterator[None]:
    """Isolate RNG state and select deterministic cuDNN kernels."""

    selected_seed = _validate_seed(seed)
    cuda_devices: list[int] = []
    if device.type == "cuda":
        cuda_devices.append(
            torch.cuda.current_device() if device.index is None else device.index
        )
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(selected_seed)
        if device.type == "cuda":
            with torch.cuda.device(cuda_devices[0]):
                torch.cuda.manual_seed(selected_seed)
        with torch.backends.cudnn.flags(benchmark=False, deterministic=True):
            yield


def _validate_stage1_pair(
    model: PCFCycleModel,
    raw_windows: Any,
    feature_vectors: Any,
    *,
    synthetic_test: bool,
    runs_per_class: int,
) -> tuple[Tensor, Tensor]:
    raw = _require_float32(raw_windows, name="stage1_raw_windows")
    features = _require_float32(feature_vectors, name="stage1_feature_vectors")
    if raw.ndim != 4 or tuple(raw.shape[2:]) != (RAW_CHANNELS, RAW_LENGTH):
        raise ValueError(
            "stage1_raw_windows must have shape [runs,endpoints,52,128]"
        )
    if features.ndim != 3 or features.shape[2] != FEATURE_DIM:
        raise ValueError(
            "stage1_feature_vectors must have shape [runs,endpoints,604]"
        )
    if tuple(raw.shape[:2]) != tuple(features.shape[:2]):
        raise ValueError("stage1 raw and feature run/endpoint axes must agree")
    if raw.shape[0] < 1 or raw.shape[1] < 1:
        raise ValueError("stage1 data must be nonempty")
    if raw.device != features.device or raw.device != _model_device(model):
        raise ValueError("stage1 tensors must already be on the model device")
    if not synthetic_test and tuple(raw.shape[:2]) != (runs_per_class, MONITOR_COUNT):
        raise ValueError(
            "production stage1 requires runs_per_class x 85 fit-healthy rows"
        )
    if not synthetic_test:
        _require_clipped_bfeat(features, name="stage1_feature_vectors")
    return raw, features


def fit_stage1(
    model: PCFCycleModel,
    healthy_raw_windows: Any,
    healthy_feature_vectors: Any,
    *,
    seed: int = BUILD_SEED,
    runs_per_class: int = 60,
    synthetic_test: bool = False,
    epoch_override: int | None = None,
) -> TrainingHistory:
    """Fit the cycle core with alternating discriminator/generator steps."""

    value = _require_model(model)
    production_runs = _require_production_runs(runs_per_class)
    epochs = _resolve_epochs(
        STAGE1_EPOCHS,
        epoch_override,
        synthetic_test=synthetic_test,
        stage="stage1",
    )
    raw, features = _validate_stage1_pair(
        value,
        healthy_raw_windows,
        healthy_feature_vectors,
        synthetic_test=synthetic_test,
        runs_per_class=production_runs,
    )
    device = raw.device
    flat_raw = raw.reshape(-1, RAW_CHANNELS, RAW_LENGTH)
    flat_features = features.reshape(-1, FEATURE_DIM)
    sample_count = flat_raw.shape[0]
    generator_parameters = [
        *value.cycle.g_ab.parameters(),
        *value.cycle.g_ba.parameters(),
    ]
    discriminator_parameters = [
        *value.cycle.d_a.parameters(),
        *value.cycle.d_b.parameters(),
    ]
    for parameter in (*generator_parameters, *discriminator_parameters):
        parameter.requires_grad_(True)
    generator_optimizer = torch.optim.AdamW(
        generator_parameters,
        lr=STAGE1_LEARNING_RATE,
        betas=STAGE1_BETAS,
        eps=STAGE1_EPSILON,
        weight_decay=STAGE1_WEIGHT_DECAY,
    )
    discriminator_optimizer = torch.optim.AdamW(
        discriminator_parameters,
        lr=STAGE1_LEARNING_RATE,
        betas=STAGE1_BETAS,
        eps=STAGE1_EPSILON,
        weight_decay=STAGE1_WEIGHT_DECAY,
    )
    rows: list[dict[str, float]] = []
    value.cycle.train()
    value.detector.eval()
    with _deterministic_training(seed, device):
        order_generator = torch.Generator(device="cpu").manual_seed(seed)
        for epoch in range(epochs):
            order = torch.randperm(sample_count, generator=order_generator)
            discriminator_sum = 0.0
            generator_sum = 0.0
            observed = 0
            batches = 0
            for start in range(0, sample_count, STAGE1_BATCH_SIZE):
                indices = order[start : start + STAGE1_BATCH_SIZE].to(device=device)
                batch_raw = flat_raw.index_select(0, indices)
                batch_features = flat_features.index_select(0, indices)
                batch_count = batch_raw.shape[0]

                discriminator_optimizer.zero_grad(set_to_none=True)
                discriminator = discriminator_lsgan_loss(
                    value.cycle, batch_raw, batch_features
                )
                discriminator.total.backward()
                discriminator_optimizer.step()
                discriminator_optimizer.zero_grad(set_to_none=True)

                generator_optimizer.zero_grad(set_to_none=True)
                generator = generator_lsgan_cycle_loss(
                    value.cycle, batch_raw, batch_features
                )
                generator.total.backward()
                generator_optimizer.step()
                generator_optimizer.zero_grad(set_to_none=True)

                discriminator_sum += (
                    float(discriminator.total.detach().item()) * batch_count
                )
                generator_sum += float(generator.total.detach().item()) * batch_count
                observed += batch_count
                batches += 1
            rows.append(
                {
                    "epoch": float(epoch + 1),
                    "discriminator_loss": discriminator_sum / observed,
                    "generator_loss": generator_sum / observed,
                    "batches": float(batches),
                }
            )
    value.cycle.eval()
    assert_finite_model_parameters(value, stage="stage1")
    return TrainingHistory(
        stage="stage1",
        seed=seed,
        configuration={
            "epochs": epochs,
            "batch_size": STAGE1_BATCH_SIZE,
            "learning_rate": STAGE1_LEARNING_RATE,
            "weight_decay": STAGE1_WEIGHT_DECAY,
            "betas": list(STAGE1_BETAS),
            "epsilon": STAGE1_EPSILON,
            "optimizer": "AdamW",
            "drop_last": False,
            "early_stopping": False,
            "runs": raw.shape[0],
            "monitor_endpoints_per_run": raw.shape[1],
            "fit_rows": sample_count,
            "production_runs_per_class": production_runs,
            "synthetic_test": synthetic_test,
        },
        epochs=tuple(rows),
    )


def _validate_signature_mapping(
    signatures_by_class: Any,
    *,
    model_device: torch.device,
    synthetic_test: bool,
    production_runs: int,
    require_two_run_batches: bool,
) -> tuple[dict[int, Tensor], int]:
    if not isinstance(signatures_by_class, Mapping):
        raise TypeError("signatures_by_class must be a mapping")
    if any(
        isinstance(key, bool) or not isinstance(key, int)
        for key in signatures_by_class
    ):
        raise ValueError("signature mapping keys must be integer class IDs")
    if set(signatures_by_class) != set(CLASS_IDS):
        raise ValueError("signature mapping must contain exactly classes 0/3/9/15")
    validated: dict[int, Tensor] = {}
    run_count: int | None = None
    for class_id in CLASS_IDS:
        tensor = _require_float32(
            signatures_by_class[class_id], name=f"class_{class_id}_signatures"
        )
        if tensor.ndim != 3 or tuple(tensor.shape[1:]) != (
            TRAIN_ENDPOINT_COUNT,
            RESIDUAL_SIGNATURE_DIM,
        ):
            raise ValueError("each signature tensor must have shape [runs,93,656]")
        if tensor.device != model_device:
            raise ValueError("all signature tensors must be on the detector device")
        if run_count is None:
            run_count = tensor.shape[0]
        elif tensor.shape[0] != run_count:
            raise ValueError("all four classes must have the same run count")
        validated[class_id] = tensor
    assert run_count is not None
    if require_two_run_batches:
        if run_count < STAGE2_RUNS_PER_CLASS_BATCH or run_count % 2:
            raise ValueError(
                "stage2 requires a positive even run count for 2/class batches"
            )
        if not synthetic_test and run_count != production_runs:
            raise ValueError(
                "production stage2 run count must equal runs_per_class"
            )
    return validated, run_count


def assemble_stage2_loss_inputs(
    detector: Stage2Detector,
    signatures_by_class: Mapping[int, Tensor],
) -> Stage2LossInputs:
    """Assemble one exact batch containing two complete runs from each class."""

    if not isinstance(detector, Stage2Detector):
        raise TypeError("detector must be a Stage2Detector")
    try:
        device = next(detector.parameters()).device
    except StopIteration as error:  # pragma: no cover
        raise RuntimeError("Stage2Detector unexpectedly has no parameters") from error
    signatures, run_count = _validate_signature_mapping(
        signatures_by_class,
        model_device=device,
        synthetic_test=True,
        production_runs=60,
        require_two_run_batches=False,
    )
    if run_count != STAGE2_RUNS_PER_CLASS_BATCH:
        raise ValueError("a stage2 loss batch must contain exactly 2 runs per class")

    healthy_monitor = signatures[0][:, MONITOR_SLICE, :]
    fault_prefix = torch.cat(
        [signatures[class_id][:, PREFIX_SLICE, :] for class_id in FAULT_IDS], dim=0
    )
    fault_mature = torch.cat(
        [signatures[class_id][:, MATURE_SLICE, :] for class_id in FAULT_IDS], dim=0
    )
    healthy_count = run_count * MONITOR_COUNT
    fault_run_count = run_count * len(FAULT_IDS)
    prefix_sample_count = fault_run_count * PREFIX_COUNT
    mature_sample_count = fault_run_count * MATURE_COUNT
    selected = torch.cat(
        (
            healthy_monitor.reshape(healthy_count, RESIDUAL_SIGNATURE_DIM),
            fault_prefix.reshape(prefix_sample_count, RESIDUAL_SIGNATURE_DIM),
            fault_mature.reshape(mature_sample_count, RESIDUAL_SIGNATURE_DIM),
        ),
        dim=0,
    )
    outputs = detector(selected)
    healthy_stop = healthy_count
    prefix_stop = healthy_stop + prefix_sample_count
    healthy_embeddings = outputs.embedding[:healthy_stop].reshape(
        run_count, MONITOR_COUNT, EMBEDDING_DIM
    )
    mature_embeddings = outputs.embedding[prefix_stop:].reshape(
        fault_run_count, MATURE_COUNT, EMBEDDING_DIM
    )
    prefix_logits = outputs.detection_logit[healthy_stop:prefix_stop].reshape(
        fault_run_count, PREFIX_COUNT
    )
    post_logits = outputs.detection_logit[prefix_stop:].reshape(
        fault_run_count, MATURE_COUNT
    )
    detection_logits = torch.cat(
        (
            outputs.detection_logit[:healthy_stop],
            outputs.detection_logit[healthy_stop:prefix_stop],
            outputs.detection_logit[prefix_stop:],
        )
    )
    binary_targets = torch.cat(
        (
            torch.zeros(
                healthy_count + prefix_sample_count,
                device=device,
                dtype=torch.long,
            ),
            torch.ones(mature_sample_count, device=device, dtype=torch.long),
        )
    )
    embeddings = torch.cat(
        (healthy_embeddings.mean(dim=1), mature_embeddings.mean(dim=1)), dim=0
    )
    contrastive_class_labels = torch.tensor(
        (0, 0, 3, 3, 9, 9, 15, 15), device=device, dtype=torch.long
    )
    ordered_fault_runs = torch.tensor(
        (3, 3, 9, 9, 15, 15), device=device, dtype=torch.long
    )
    attribution_fault_ids = ordered_fault_runs.repeat_interleave(MATURE_COUNT)
    attribution_logits = outputs.attribution_logits[prefix_stop:]
    return Stage2LossInputs(
        detection_logits=detection_logits,
        binary_targets=binary_targets,
        post_logits=post_logits,
        prefix_logits=prefix_logits,
        embeddings=embeddings,
        contrastive_class_labels=contrastive_class_labels,
        attribution_logits=attribution_logits,
        attribution_fault_ids=attribution_fault_ids,
    )


def fit_stage2(
    model: PCFCycleModel,
    signatures_by_class: Mapping[int, Tensor],
    *,
    seed: int = BUILD_SEED,
    runs_per_class: int = 60,
    synthetic_test: bool = False,
    epoch_override: int | None = None,
) -> TrainingHistory:
    """Train only the detector in batches of exactly two runs per class."""

    value = _require_model(model)
    production_runs = _require_production_runs(runs_per_class)
    epochs = _resolve_epochs(
        STAGE2_EPOCHS,
        epoch_override,
        synthetic_test=synthetic_test,
        stage="stage2",
    )
    device = _model_device(value)
    signatures, run_count = _validate_signature_mapping(
        signatures_by_class,
        model_device=device,
        synthetic_test=synthetic_test,
        production_runs=production_runs,
        require_two_run_batches=True,
    )
    for parameter in value.cycle.parameters():
        parameter.requires_grad_(False)
    for parameter in value.detector.parameters():
        parameter.requires_grad_(True)
    value.cycle.eval()
    value.detector.train()
    optimizer = torch.optim.AdamW(
        value.detector.parameters(),
        lr=STAGE2_LEARNING_RATE,
        betas=STAGE2_BETAS,
        eps=STAGE2_EPSILON,
        weight_decay=STAGE2_WEIGHT_DECAY,
    )
    metric_names = (
        "total",
        "balanced_bce",
        "paired_rank",
        "supervised_contrastive",
        "mature_fault_attribution",
    )
    rows: list[dict[str, float]] = []
    with _deterministic_training(seed, device):
        order_generator = torch.Generator(device="cpu").manual_seed(seed)
        for epoch in range(epochs):
            orders = {
                class_id: torch.randperm(run_count, generator=order_generator)
                for class_id in CLASS_IDS
            }
            sums = {name: 0.0 for name in metric_names}
            batches = run_count // STAGE2_RUNS_PER_CLASS_BATCH
            for batch_index in range(batches):
                start = batch_index * STAGE2_RUNS_PER_CLASS_BATCH
                stop = start + STAGE2_RUNS_PER_CLASS_BATCH
                batch = {
                    class_id: signatures[class_id].index_select(
                        0, orders[class_id][start:stop].to(device=device)
                    )
                    for class_id in CLASS_IDS
                }
                optimizer.zero_grad(set_to_none=True)
                loss = assemble_stage2_loss_inputs(value.detector, batch).loss()
                loss.total.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                for name in metric_names:
                    tensor = loss.total if name == "total" else loss.components[name]
                    sums[name] += float(tensor.detach().item())
            row = {name: sums[name] / batches for name in metric_names}
            row["epoch"] = float(epoch + 1)
            row["batches"] = float(batches)
            rows.append(row)
    value.eval()
    assert_finite_model_parameters(value, stage="stage2")
    return TrainingHistory(
        stage="stage2",
        seed=seed,
        configuration={
            "epochs": epochs,
            "learning_rate": STAGE2_LEARNING_RATE,
            "weight_decay": STAGE2_WEIGHT_DECAY,
            "betas": list(STAGE2_BETAS),
            "epsilon": STAGE2_EPSILON,
            "optimizer": "AdamW",
            "drop_last": False,
            "early_stopping": False,
            "runs_per_class": run_count,
            "production_runs_per_class": production_runs,
            "runs_per_class_batch": STAGE2_RUNS_PER_CLASS_BATCH,
            "batches_per_epoch": batches,
            "prefix_endpoints": list(PREFIX_ENDPOINTS),
            "monitor_endpoints": list(MONITOR_TRAIN_ENDPOINTS),
            "mature_fault_endpoints": list(MATURE_FAULT_ENDPOINTS),
            "ignored_fault_endpoints": list(TRAIN_ENDPOINTS[IGNORED_FAULT_SLICE]),
            "synthetic_test": synthetic_test,
        },
        epochs=tuple(rows),
    )


def precompute_residual_signatures(
    model: PCFCycleModel,
    raw_windows: Any,
    feature_vectors: Any,
    *,
    batch_size: int = PRECOMPUTE_BATCH_SIZE,
) -> Tensor:
    """Precompute frozen-cycle signatures at any nonempty endpoint axis."""

    value = _require_model(model)
    size = _require_positive_integer(batch_size, name="batch_size")
    raw = _require_float32(raw_windows, name="raw_windows")
    features = _require_float32(feature_vectors, name="feature_vectors")
    _require_clipped_bfeat(features, name="feature_vectors")
    if raw.ndim != 4 or tuple(raw.shape[2:]) != (RAW_CHANNELS, RAW_LENGTH):
        raise ValueError("raw_windows must have shape [runs,endpoints,52,128]")
    if features.ndim != 3 or features.shape[2] != FEATURE_DIM:
        raise ValueError("feature_vectors must have shape [runs,endpoints,604]")
    if tuple(raw.shape[:2]) != tuple(features.shape[:2]):
        raise ValueError("raw and feature run/endpoint axes must agree")
    if raw.shape[0] < 1 or raw.shape[1] < 1:
        raise ValueError("raw and feature endpoint tensors must be nonempty")
    device = _model_device(value)
    if raw.device != features.device or raw.device != device:
        raise ValueError("endpoint tensors must already be on the model device")
    for parameter in value.cycle.parameters():
        parameter.requires_grad_(False)
    value.cycle.eval()
    run_count, endpoint_count = raw.shape[:2]
    flat_raw = raw.reshape(-1, RAW_CHANNELS, RAW_LENGTH)
    flat_features = features.reshape(-1, FEATURE_DIM)
    output: list[Tensor] = []
    with torch.no_grad():
        for start in range(0, flat_raw.shape[0], size):
            batch_raw = flat_raw[start : start + size]
            batch_features = flat_features[start : start + size]
            cycle = value.cycle(batch_raw, batch_features)
            output.append(
                residual_signature(batch_raw, batch_features, cycle)
            )
    return torch.cat(output).reshape(
        run_count, endpoint_count, RESIDUAL_SIGNATURE_DIM
    )


def precompute_stage2_signatures(
    model: PCFCycleModel,
    raw_by_class: Mapping[int, Tensor],
    features_by_class: Mapping[int, Tensor],
    *,
    batch_size: int = PRECOMPUTE_BATCH_SIZE,
    runs_per_class: int = 60,
    synthetic_test: bool = False,
) -> dict[int, Tensor]:
    """Precompute the 8 prefix plus 85 monitor signatures for four classes."""

    value = _require_model(model)
    production_runs = _require_production_runs(runs_per_class)
    if not isinstance(synthetic_test, bool):
        raise TypeError("synthetic_test must be boolean")
    if not isinstance(raw_by_class, Mapping) or not isinstance(
        features_by_class, Mapping
    ):
        raise TypeError("raw_by_class and features_by_class must be mappings")
    if set(raw_by_class) != set(CLASS_IDS) or set(features_by_class) != set(CLASS_IDS):
        raise ValueError("stage2 raw/features must contain exactly classes 0/3/9/15")
    paired: dict[int, tuple[Tensor, Tensor]] = {}
    common_runs: int | None = None
    for class_id in CLASS_IDS:
        raw = _require_float32(raw_by_class[class_id], name=f"class_{class_id}_raw")
        features = _require_float32(
            features_by_class[class_id], name=f"class_{class_id}_features"
        )
        if raw.ndim != 4 or tuple(raw.shape[1:]) != (
            TRAIN_ENDPOINT_COUNT,
            RAW_CHANNELS,
            RAW_LENGTH,
        ):
            raise ValueError("each stage2 raw tensor must have shape [runs,93,52,128]")
        if features.ndim != 3 or tuple(features.shape[1:]) != (
            TRAIN_ENDPOINT_COUNT,
            FEATURE_DIM,
        ):
            raise ValueError(
                "each stage2 feature tensor must have shape [runs,93,604]"
            )
        if raw.shape[0] != features.shape[0]:
            raise ValueError("stage2 paired run counts must agree")
        if not synthetic_test:
            _require_clipped_bfeat(
                features, name=f"class_{class_id}_features"
            )
        if common_runs is None:
            common_runs = raw.shape[0]
        elif raw.shape[0] != common_runs:
            raise ValueError("all stage2 classes must have the same run count")
        paired[class_id] = (raw, features)
    assert common_runs is not None
    if common_runs < 2 or common_runs % 2:
        raise ValueError("stage2 requires a positive even run count")
    if not synthetic_test and common_runs != production_runs:
        raise ValueError("production stage2 run count must equal runs_per_class")
    return {
        class_id: precompute_residual_signatures(
            value, raw, features, batch_size=batch_size
        )
        for class_id, (raw, features) in paired.items()
    }


def infer_endpoint_scores(
    model: PCFCycleModel,
    raw_windows: Any,
    feature_vectors: Any,
    *,
    batch_size: int = PRECOMPUTE_BATCH_SIZE,
) -> EndpointScores:
    """Run batched inference over any endpoint axis on the model device."""

    features = _require_float32(feature_vectors, name="feature_vectors")
    _require_clipped_bfeat(features, name="feature_vectors")
    return score_endpoint_pairs(
        _require_model(model),
        raw_windows,
        features,
        batch_size=batch_size,
    )


def train_pcfcycle(
    healthy_raw_windows: Tensor,
    healthy_feature_vectors: Tensor,
    stage2_raw_by_class: Mapping[int, Tensor],
    stage2_features_by_class: Mapping[int, Tensor],
    *,
    seed: int = BUILD_SEED,
    runs_per_class: int = 60,
    synthetic_test: bool = False,
    stage1_epoch_override: int | None = None,
    stage2_epoch_override: int | None = None,
    precompute_batch_size: int = PRECOMPUTE_BATCH_SIZE,
    checkpoint_path: str | Path | None = None,
) -> PCFCycleTrainingResult:
    """Run both stages on inputs that already share one CPU or CUDA device."""

    raw = _require_float32(healthy_raw_windows, name="healthy_raw_windows")
    selected_seed = _validate_seed(seed)
    production_runs = _require_production_runs(runs_per_class)
    model = build_pcfcycle_model(seed=selected_seed, device=raw.device)
    stage1 = fit_stage1(
        model,
        healthy_raw_windows,
        healthy_feature_vectors,
        seed=selected_seed,
        runs_per_class=production_runs,
        synthetic_test=synthetic_test,
        epoch_override=stage1_epoch_override,
    )
    signatures = precompute_stage2_signatures(
        model,
        stage2_raw_by_class,
        stage2_features_by_class,
        batch_size=precompute_batch_size,
        runs_per_class=production_runs,
        synthetic_test=synthetic_test,
    )
    stage2 = fit_stage2(
        model,
        signatures,
        seed=selected_seed,
        runs_per_class=production_runs,
        synthetic_test=synthetic_test,
        epoch_override=stage2_epoch_override,
    )
    result = PCFCycleTrainingResult(model, stage1, stage2)
    if checkpoint_path is not None:
        result.save(checkpoint_path)
    return result


def load_training_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> PCFCycleTrainingResult:
    """Load a complete result and fail closed on training metadata drift."""

    source = Path(path)
    document = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(document, Mapping):
        raise ValueError("training checkpoint document is invalid")
    metadata = document.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("training checkpoint metadata is missing")
    if metadata.get("candidate_id") != CANDIDATE_ID:
        raise ValueError("training checkpoint candidate ID changed")
    if metadata.get("training_schema") != TRAINING_SCHEMA:
        raise ValueError("training checkpoint schema changed")
    stage1 = TrainingHistory.from_metadata(
        metadata.get("stage1_history"), expected_stage="stage1"
    )
    stage2 = TrainingHistory.from_metadata(
        metadata.get("stage2_history"), expected_stage="stage2"
    )
    if document.get("build_seed") != stage1.seed or stage1.seed != stage2.seed:
        raise ValueError("training checkpoint build/training seed changed")
    model = load_pcfcycle_checkpoint(source, device=device)
    assert_finite_model_parameters(model, stage="checkpoint load")
    model.eval()
    for parameter in model.cycle.parameters():
        parameter.requires_grad_(False)
    result = PCFCycleTrainingResult(model, stage1, stage2)
    expected = result.metadata()
    for key in (
        "runs_per_class",
        "synthetic_test",
        "bfeat_normalization",
    ):
        if metadata.get(key) != expected[key]:
            raise ValueError(f"training checkpoint {key} contract changed")
    return result


__all__ = [
    "ALLOWED_PRODUCTION_RUNS",
    "BFEAT_CLIP_BOUND",
    "BFEAT_MAD_CONSISTENCY",
    "BFEAT_NORMALIZATION",
    "BFEAT_SCALE_FLOOR",
    "CANDIDATE_ID",
    "CLASS_IDS",
    "IGNORED_FAULT_SLICE",
    "MATURE_SLICE",
    "MONITOR_SLICE",
    "PCFCycleTrainingResult",
    "PREFIX_SLICE",
    "PRECOMPUTE_BATCH_SIZE",
    "STAGE1_BATCH_SIZE",
    "STAGE1_BETAS",
    "STAGE1_EPSILON",
    "STAGE1_EPOCHS",
    "STAGE1_LEARNING_RATE",
    "STAGE1_WEIGHT_DECAY",
    "STAGE2_BETAS",
    "STAGE2_EPSILON",
    "STAGE2_EPOCHS",
    "STAGE2_LEARNING_RATE",
    "STAGE2_RUNS_PER_CLASS_BATCH",
    "STAGE2_WEIGHT_DECAY",
    "Stage2LossInputs",
    "TRAINING_SCHEMA",
    "TrainingHistory",
    "assemble_stage2_loss_inputs",
    "assert_finite_model_parameters",
    "bfeat_normalization_contract",
    "fit_stage1",
    "fit_stage2",
    "infer_endpoint_scores",
    "load_training_checkpoint",
    "precompute_residual_signatures",
    "precompute_stage2_signatures",
    "train_pcfcycle",
]
