"""Development and healthy-calibration-only CPU/CUDA parity diagnosis."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch  # Import before NumPy on Windows so CUDA DLL resolution is stable.
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_hydraulic_padr_pac_gpu as runner
from mate_spc_hydraulic.data import load_hydraulic_cycles, load_phasewise_template
from mate_spc_hydraulic.model import load_hydraulic_model, score_hydraulic_cuda, score_hydraulic_numpy


def main() -> int:
    root = PROJECT_ROOT
    args = argparse.Namespace(
        data_dir=root / "data" / "hydraulic_uci_447",
        output_dir=root / "outputs" / "hydraulic_padr_pac_gpu_v1_1",
        phase="development",
        device="cuda",
        batch_size=64,
        analysis_protocol=None,
        analysis_protocol_sidecar=None,
        parent_analysis_protocol=None,
        parent_analysis_protocol_sidecar=None,
        aborted_parent_record=None,
        acquisition_protocol=None,
        acquisition_protocol_sidecar=None,
        file_manifest=None,
        file_manifest_sidecar=None,
        label_audit_dir=None,
        archive=None,
    )
    runner._resolve_paths(args, root)
    preflight = runner._preflight(args, root)
    development = preflight.records["development"]
    calibration = tuple(
        record
        for record in preflight.records["calibration"]
        if record.valve_label == 100
    )
    requested = development + calibration
    indices, _, _, _ = runner._records_arrays(requested)
    raw, _ = load_hydraulic_cycles(args.data_dir, indices.tolist())
    template = load_phasewise_template(args.output_dir / "phase_template.npz")
    model, metadata = load_hydraulic_model(args.output_dir / "calibrated_model.npz")
    features = template.transform(raw)
    cpu = score_hydraulic_numpy(model, features)
    gpu, execution = score_hydraulic_cuda(model, features, device="cuda", batch_size=64)

    names = (
        "logits",
        "standardized_logits",
        "clipped_evidence",
        "cusum",
        "component_persistence",
        "envelope",
    )
    errors = {}
    for name in names:
        left = np.asarray(getattr(cpu, name))
        right = np.asarray(getattr(gpu, name))
        finite = np.isfinite(left) & np.isfinite(right)
        errors[name] = float(np.max(np.abs(left[finite] - right[finite])))
    mismatch = np.argwhere(cpu.alarm != gpu.alarm)
    print({"threshold": model.threshold, "selected_lambda": metadata["selected_ridge_lambda"]})
    print({"numeric_max_abs_errors": errors})
    print({"eligible_exact": bool(np.array_equal(cpu.eligible, gpu.eligible)), "alarm_mismatch_count": int(len(mismatch))})
    for cycle, phase in mismatch[:20]:
        record = requested[int(cycle)]
        print(
            {
                "requested_offset": int(cycle),
                "cycle_index_zero_based": record.index,
                "split": record.split,
                "phase_zero_based": int(phase),
                "cpu_envelope": float(cpu.envelope[cycle, phase]),
                "gpu_envelope": float(gpu.envelope[cycle, phase]),
                "cpu_alarm": bool(cpu.alarm[cycle, phase]),
                "gpu_alarm": bool(gpu.alarm[cycle, phase]),
            }
        )

    start = len(development)
    cpu_maxima = np.max(cpu.envelope[start:, 2:], axis=1)
    gpu_maxima = np.max(gpu.envelope[start:, 2:], axis=1)
    cpu_driver = int(np.argmax(cpu_maxima))
    gpu_driver = int(np.argmax(gpu_maxima))
    print(
        {
            "cpu_calibration_maximum": float(cpu_maxima[cpu_driver]),
            "cpu_driver_cycle": calibration[cpu_driver].index,
            "gpu_calibration_maximum": float(gpu_maxima[gpu_driver]),
            "gpu_driver_cycle": calibration[gpu_driver].index,
            "gpu_minus_cpu_at_cpu_driver": float(gpu_maxima[cpu_driver] - cpu_maxima[cpu_driver]),
            "gpu_max_minus_threshold": float(np.max(gpu_maxima) - model.threshold),
            "device": execution["device_name"],
        }
    )
    for guard in (0.0, 1.0e-12, 1.0e-11, 1.0e-10, 1.0e-9):
        boundary = float(np.nextafter(model.threshold + guard, np.inf))
        cpu_alarm = cpu.eligible & (cpu.envelope > boundary)
        gpu_alarm = gpu.eligible & (gpu.envelope > boundary)
        print(
            {
                "guard": guard,
                "boundary": boundary,
                "alarm_mismatch_count": int(np.count_nonzero(cpu_alarm != gpu_alarm)),
                "cpu_calibration_alarm_cycles": int(np.count_nonzero(np.any(cpu_alarm[start:], axis=1))),
                "gpu_calibration_alarm_cycles": int(np.count_nonzero(np.any(gpu_alarm[start:], axis=1))),
            }
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
