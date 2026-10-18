from __future__ import annotations

from pathlib import Path

import pytest
import torch
import mate_spc_pcfcycle.training as training_module

from mate_spc_pcfcycle.model import (
    BUILD_SEED,
    build_pcfcycle_model,
    state_sha256,
)
from mate_spc_pcfcycle.training import (
    ALLOWED_PRODUCTION_RUNS,
    CANDIDATE_ID,
    CLASS_IDS,
    PCFCycleTrainingResult,
    STAGE1_BATCH_SIZE,
    STAGE1_BETAS,
    STAGE1_EPOCHS,
    STAGE1_EPSILON,
    STAGE1_LEARNING_RATE,
    STAGE1_WEIGHT_DECAY,
    STAGE2_EPOCHS,
    STAGE2_EPSILON,
    STAGE2_BETAS,
    STAGE2_LEARNING_RATE,
    STAGE2_RUNS_PER_CLASS_BATCH,
    STAGE2_WEIGHT_DECAY,
    TrainingHistory,
    assemble_stage2_loss_inputs,
    assert_finite_model_parameters,
    bfeat_normalization_contract,
    fit_stage1,
    fit_stage2,
    infer_endpoint_scores,
    load_training_checkpoint,
    precompute_residual_signatures,
)


def _device_or_skip(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return torch.device(name)


def _stage1_pair(
    *, device: torch.device, runs: int = 2, endpoints: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(20260912)
    raw = torch.randn(
        (runs, endpoints, 52, 128), generator=generator, device=device
    )
    features = torch.randn(
        (runs, endpoints, 604), generator=generator, device=device
    )
    return raw, features


def _signatures(
    *, device: torch.device, runs: int = 2
) -> dict[int, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(20260913)
    return {
        class_id: torch.randn(
            (runs, 93, 656), generator=generator, device=device
        )
        for class_id in CLASS_IDS
    }


def _clone_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in module.state_dict().items()}


def _assert_state_equal(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> None:
    assert left.keys() == right.keys()
    for name in left:
        torch.testing.assert_close(left[name], right[name], rtol=0.0, atol=0.0)


def _history_pair(
    *,
    runs: int = 2,
    synthetic: bool = True,
    production_runs: int = 60,
) -> tuple[TrainingHistory, TrainingHistory]:
    stage1_epochs = 1 if synthetic else STAGE1_EPOCHS
    stage2_epochs = 1 if synthetic else STAGE2_EPOCHS
    monitor_count = 1 if synthetic else 85
    fit_rows = runs * monitor_count
    stage1_batches = (fit_rows + STAGE1_BATCH_SIZE - 1) // STAGE1_BATCH_SIZE
    stage2_batches = runs // STAGE2_RUNS_PER_CLASS_BATCH
    stage1 = TrainingHistory(
        "stage1",
        BUILD_SEED,
        {
            "epochs": stage1_epochs,
            "batch_size": STAGE1_BATCH_SIZE,
            "learning_rate": STAGE1_LEARNING_RATE,
            "weight_decay": STAGE1_WEIGHT_DECAY,
            "betas": list(STAGE1_BETAS),
            "epsilon": STAGE1_EPSILON,
            "optimizer": "AdamW",
            "drop_last": False,
            "early_stopping": False,
            "runs": runs,
            "monitor_endpoints_per_run": monitor_count,
            "fit_rows": fit_rows,
            "production_runs_per_class": production_runs,
            "synthetic_test": synthetic,
        },
        tuple(
            {
                "epoch": float(epoch),
                "discriminator_loss": 2.0,
                "generator_loss": 3.0,
                "batches": float(stage1_batches),
            }
            for epoch in range(1, stage1_epochs + 1)
        ),
    )
    stage2 = TrainingHistory(
        "stage2",
        BUILD_SEED,
        {
            "epochs": stage2_epochs,
            "learning_rate": STAGE2_LEARNING_RATE,
            "weight_decay": STAGE2_WEIGHT_DECAY,
            "betas": list(STAGE2_BETAS),
            "epsilon": STAGE2_EPSILON,
            "optimizer": "AdamW",
            "drop_last": False,
            "early_stopping": False,
            "runs_per_class": runs,
            "production_runs_per_class": production_runs,
            "runs_per_class_batch": STAGE2_RUNS_PER_CLASS_BATCH,
            "batches_per_epoch": stage2_batches,
            "prefix_endpoints": list(range(128, 157, 4)),
            "monitor_endpoints": list(range(160, 497, 4)),
            "mature_fault_endpoints": list(range(192, 497, 4)),
            "ignored_fault_endpoints": list(range(160, 189, 4)),
            "synthetic_test": synthetic,
        },
        tuple(
            {
                "epoch": float(epoch),
                "total": 1.0,
                "balanced_bce": 1.0,
                "paired_rank": 1.0,
                "supervised_contrastive": 1.0,
                "mature_fault_attribution": 1.0,
                "batches": float(stage2_batches),
            }
            for epoch in range(1, stage2_epochs + 1)
        ),
    )
    return stage1, stage2


def test_frozen_production_constants_normalization_and_epoch_guards() -> None:
    assert CANDIDATE_ID == "pcf-cyclerank-scan-v6-fixed"
    assert ALLOWED_PRODUCTION_RUNS == (60, 100)
    assert (STAGE1_EPOCHS, STAGE1_BATCH_SIZE) == (120, 128)
    assert STAGE1_BETAS == (0.9, 0.999)
    assert STAGE1_EPSILON == 1.0e-8
    assert (STAGE1_LEARNING_RATE, STAGE1_WEIGHT_DECAY) == (1.0e-4, 1.0e-4)
    assert (STAGE2_EPOCHS, STAGE2_RUNS_PER_CLASS_BATCH) == (80, 2)
    assert STAGE2_BETAS == (0.9, 0.999)
    assert STAGE2_EPSILON == 1.0e-8
    assert (STAGE2_LEARNING_RATE, STAGE2_WEIGHT_DECAY) == (3.0e-4, 1.0e-4)
    assert bfeat_normalization_contract(60)["fit_rows"] == 60 * 85
    assert bfeat_normalization_contract(100)["fit_rows"] == 100 * 85
    assert bfeat_normalization_contract(60)["scale_floor"] == 1.0e-6
    assert bfeat_normalization_contract(60)["mad_consistency"] == 1.4826
    assert bfeat_normalization_contract(60)["scale"].startswith("1.4826")
    assert bfeat_normalization_contract(60)["clip"] == [-10.0, 10.0]
    with pytest.raises(ValueError, match="60 or 100"):
        bfeat_normalization_contract(80)
    with pytest.raises(ValueError, match="integer"):
        bfeat_normalization_contract(60.0)  # type: ignore[arg-type]

    model = build_pcfcycle_model()
    raw, features = _stage1_pair(device=torch.device("cpu"))
    with pytest.raises(ValueError, match="only for synthetic tests"):
        fit_stage1(model, raw, features, epoch_override=1)
    with pytest.raises(ValueError, match="runs_per_class"):
        fit_stage1(
            model,
            raw,
            features,
            runs_per_class=80,
            synthetic_test=True,
            epoch_override=1,
        )
    with pytest.raises(ValueError, match="runs_per_class x 85"):
        fit_stage1(model, raw, features, runs_per_class=100)
    with pytest.raises(ValueError, match="only for synthetic tests"):
        fit_stage2(model, _signatures(device=torch.device("cpu")), epoch_override=1)


def test_exact_stage2_populations_and_ignored_fault_region() -> None:
    detector = build_pcfcycle_model().detector.eval()
    signatures = _signatures(device=torch.device("cpu"))
    with torch.no_grad():
        observed = assemble_stage2_loss_inputs(detector, signatures)
    assert observed.detection_logits.shape == (680,)
    assert observed.binary_targets.shape == (680,)
    assert int((observed.binary_targets == 0).sum()) == 2 * 85 + 6 * 8
    assert int((observed.binary_targets == 1).sum()) == 6 * 77
    assert observed.prefix_logits.shape == (6, 8)
    assert observed.post_logits.shape == (6, 77)
    assert observed.embeddings.shape == (8, 32)
    assert observed.attribution_logits.shape == (6 * 77, 3)
    assert observed.contrastive_class_labels.tolist() == [0, 0, 3, 3, 9, 9, 15, 15]
    assert observed.attribution_fault_ids.reshape(6, 77)[:, 0].tolist() == [
        3,
        3,
        9,
        9,
        15,
        15,
    ]

    altered = {key: value.clone() for key, value in signatures.items()}
    altered[0][:, :8] += 10_000.0
    for class_id in (3, 9, 15):
        altered[class_id][:, 8:16] -= 10_000.0
    with torch.no_grad():
        ignored_changed = assemble_stage2_loss_inputs(detector, altered)
    for field in observed.__dataclass_fields__:
        torch.testing.assert_close(
            getattr(observed, field),
            getattr(ignored_changed, field),
            rtol=0.0,
            atol=0.0,
        )


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_stage1_one_epoch_is_deterministic_and_generator_discriminator_only(
    device_name: str,
) -> None:
    device = _device_or_skip(device_name)
    raw, features = _stage1_pair(device=device)
    first = build_pcfcycle_model(device=device)
    second = build_pcfcycle_model(device=device)
    initial_cycle = _clone_state(first.cycle)
    initial_detector = _clone_state(first.detector)
    first_history = fit_stage1(
        first,
        raw,
        features,
        synthetic_test=True,
        epoch_override=1,
    )
    second_history = fit_stage1(
        second,
        raw,
        features,
        synthetic_test=True,
        epoch_override=1,
    )
    _assert_state_equal(_clone_state(first.cycle), _clone_state(second.cycle))
    _assert_state_equal(_clone_state(first.detector), initial_detector)
    assert any(
        not torch.equal(tensor, first.cycle.state_dict()[name])
        for name, tensor in initial_cycle.items()
    )
    assert first_history.to_metadata() == second_history.to_metadata()
    assert first_history.configuration["epochs"] == 1
    assert first_history.configuration["batch_size"] == 128
    assert first_history.configuration["runs"] == 2
    assert first_history.configuration["fit_rows"] == 2
    assert len(first_history.epochs) == 1
    assert all(torch.isfinite(torch.tensor(list(first_history.epochs[0].values()))))


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_stage2_one_epoch_is_deterministic_and_keeps_cycle_frozen(
    device_name: str,
) -> None:
    device = _device_or_skip(device_name)
    signatures = _signatures(device=device)
    first = build_pcfcycle_model(device=device)
    second = build_pcfcycle_model(device=device)
    initial_cycle = _clone_state(first.cycle)
    initial_detector = _clone_state(first.detector)
    first_history = fit_stage2(
        first,
        signatures,
        synthetic_test=True,
        epoch_override=1,
    )
    second_history = fit_stage2(
        second,
        signatures,
        synthetic_test=True,
        epoch_override=1,
    )
    _assert_state_equal(_clone_state(first.cycle), initial_cycle)
    _assert_state_equal(_clone_state(first.detector), _clone_state(second.detector))
    assert any(
        not torch.equal(tensor, first.detector.state_dict()[name])
        for name, tensor in initial_detector.items()
    )
    assert all(not parameter.requires_grad for parameter in first.cycle.parameters())
    assert not first.training
    assert first_history.to_metadata() == second_history.to_metadata()
    assert first_history.configuration["epochs"] == 1
    assert first_history.configuration["runs_per_class_batch"] == 2
    assert first_history.configuration["batches_per_epoch"] == 1
    assert first_history.configuration["ignored_fault_endpoints"] == list(
        range(160, 189, 4)
    )


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_precompute_and_inference_accept_arbitrary_endpoint_count(
    device_name: str,
) -> None:
    device = _device_or_skip(device_name)
    model = build_pcfcycle_model(device=device).eval()
    raw, features = _stage1_pair(device=device, runs=1, endpoints=3)
    signatures = precompute_residual_signatures(
        model, raw, features, batch_size=2
    )
    assert signatures.shape == (1, 3, 656)
    assert signatures.device.type == device.type
    assert all(not parameter.requires_grad for parameter in model.cycle.parameters())
    with torch.no_grad():
        direct = model(
            raw.reshape(3, 52, 128), features.reshape(3, 604)
        ).residual_signature.reshape(1, 3, 656)
    torch.testing.assert_close(signatures, direct, rtol=2.0e-3, atol=5.0e-5)
    scores = infer_endpoint_scores(model, raw, features, batch_size=2)
    assert scores.embedding.shape == (1, 3, 32)
    assert scores.detection_logits.shape == (1, 3)
    assert scores.attribution_logits.shape == (1, 3, 3)
    assert scores.detection_logits.device.type == device.type


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_training_checkpoint_metadata_and_same_device_replay(
    tmp_path: Path,
    device_name: str,
) -> None:
    device = _device_or_skip(device_name)
    model = build_pcfcycle_model(device=device).eval()
    stage1, stage2 = _history_pair()
    result = PCFCycleTrainingResult(model, stage1, stage2)
    path = tmp_path / f"training_{device_name}.pt"
    manifest = result.save(path, extra_metadata={"purpose": "synthetic-test"})
    assert manifest["metadata"]["candidate_id"] == CANDIDATE_ID
    assert manifest["metadata"]["runs_per_class"] == 2
    assert manifest["metadata"]["synthetic_test"] is True
    loaded = load_training_checkpoint(path, device=device)
    assert state_sha256(loaded.model) == state_sha256(model)
    assert loaded.stage1_history.to_metadata() == stage1.to_metadata()
    assert loaded.stage2_history.to_metadata() == stage2.to_metadata()
    raw, features = _stage1_pair(device=device, runs=1, endpoints=2)
    expected = infer_endpoint_scores(model, raw, features, batch_size=1)
    actual = infer_endpoint_scores(loaded.model, raw, features, batch_size=1)
    torch.testing.assert_close(
        actual.detection_logits, expected.detection_logits, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        actual.attribution_logits, expected.attribution_logits, rtol=0.0, atol=0.0
    )


def test_final_refit_metadata_binds_one_hundred_runs() -> None:
    model = build_pcfcycle_model()
    stage1, stage2 = _history_pair(
        runs=100, synthetic=False, production_runs=100
    )
    metadata = PCFCycleTrainingResult(model, stage1, stage2).metadata()
    assert metadata["runs_per_class"] == 100
    assert metadata["bfeat_normalization"]["fit_rows"] == 8_500
    assert metadata["bfeat_normalization"]["fit_runs"] == 100


def test_checkpoint_rejects_optimizer_epsilon_tampering(tmp_path: Path) -> None:
    stage1, stage2 = _history_pair()
    path = tmp_path / "tampered-epsilon.pt"
    PCFCycleTrainingResult(build_pcfcycle_model(), stage1, stage2).save(path)
    document = torch.load(path, map_location="cpu", weights_only=False)
    document["metadata"]["stage1_history"]["configuration"]["epsilon"] = 1.0e-7
    torch.save(document, path)
    with pytest.raises(ValueError, match="stage1 optimizer/batch"):
        load_training_checkpoint(path)


def test_production_history_cannot_be_empty_or_truncated() -> None:
    stage1, stage2 = _history_pair(
        runs=60, synthetic=False, production_runs=60
    )
    truncated = TrainingHistory(
        stage1.stage,
        stage1.seed,
        stage1.configuration,
        stage1.epochs[:-1],
    )
    with pytest.raises(ValueError, match="exactly 120 epochs"):
        PCFCycleTrainingResult(build_pcfcycle_model(), truncated, stage2).metadata()


def test_finite_parameter_audit_and_bfeat_clip_guard_fail_closed() -> None:
    model = build_pcfcycle_model()
    with torch.no_grad():
        next(model.parameters()).view(-1)[0] = torch.nan
    with pytest.raises(FloatingPointError, match="nonfinite parameters"):
        assert_finite_model_parameters(model, stage="test")
    with pytest.raises(ValueError, match=r"\[-10,10\]"):
        training_module._require_clipped_bfeat(
            torch.tensor([[10.01]], dtype=torch.float32), name="features"
        )


def test_public_inference_rejects_out_of_contract_bfeat() -> None:
    model = build_pcfcycle_model()
    raw, features = _stage1_pair(device=torch.device("cpu"), runs=1, endpoints=1)
    features[0, 0, 0] = 10.01
    with pytest.raises(ValueError, match=r"\[-10,10\]"):
        precompute_residual_signatures(model, raw, features)
    with pytest.raises(ValueError, match=r"\[-10,10\]"):
        infer_endpoint_scores(model, raw, features)
