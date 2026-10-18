"""Development-only CUDA diagnostic for the aborted Hydraulic v1 optimizer cap."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch  # Import before NumPy-backed project modules on Windows.
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_hydraulic_padr_pac_gpu as runner
from mate_spc_hydraulic.data import fit_phasewise_template, load_hydraulic_cycles
from mate_spc_hydraulic.model import (
    fit_hydraulic_directional_cuda,
    protocol_grouped_fold_ids,
)


def main() -> int:
    root = PROJECT_ROOT
    args = argparse.Namespace(
        data_dir=root / "data" / "hydraulic_uci_447",
        output_dir=root / "outputs" / "_unused_optimizer_diagnostic",
        phase="development",
        device="cuda",
        batch_size=64,
        analysis_protocol=None,
        analysis_protocol_sidecar=None,
        acquisition_protocol=None,
        acquisition_protocol_sidecar=None,
        file_manifest=None,
        file_manifest_sidecar=None,
        label_audit_dir=None,
        archive=None,
    )
    runner._resolve_paths(args, root)
    preflight = runner._preflight(args, root)
    records = preflight.records["development"]
    indices, labels, groups, _ = runner._records_arrays(records)
    raw, _ = load_hydraulic_cycles(args.data_dir, indices.tolist())
    template = fit_phasewise_template(raw[labels == 100])
    features = template.transform(raw)
    folds = protocol_grouped_fold_ids(groups)
    training = folds != 0

    fitted = []
    for budget in (500, 2000):
        print(f"START max_iter={budget}", flush=True)
        model, info = fit_hydraulic_directional_cuda(
            features[training],
            labels[training],
            groups[training],
            ridge_lambda=1.0e-4,
            device="cuda",
            max_iter=budget,
            max_eval=int(budget * 1.25),
            tolerance_grad=1.0e-7,
            tolerance_change=1.0e-10,
            history_size=100,
            require_convergence=False,
        )
        keys = (
            "optimizer_iterations",
            "optimizer_function_evaluations",
            "termination_reason",
            "converged",
            "gradient_inf_norm",
            "final_objective",
            "cuda_event_elapsed_seconds",
        )
        print({key: info[key] for key in keys}, flush=True)
        fitted.append(model)

    print(
        {
            "max_abs_weight_delta_500_vs_2000": float(
                np.max(np.abs(fitted[0].weights - fitted[1].weights))
            ),
            "max_abs_bias_delta_500_vs_2000": float(
                np.max(np.abs(fitted[0].bias - fitted[1].bias))
            ),
            "device": torch.cuda.get_device_name(0),
        },
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
