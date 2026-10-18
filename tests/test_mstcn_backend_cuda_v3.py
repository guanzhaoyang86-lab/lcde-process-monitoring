from __future__ import annotations

import torch  # Windows DLL order: Torch must precede NumPy/backend.
import numpy as np
import pytest

from mate_spc_mstcn import backend_v3 as backend
from mate_spc_mstcn.features import RobustRunTransform
from mate_spc_mstcn.model import MSTCNModel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_backend_sequence_efficient_dense_cpu_cuda_parity() -> None:
    rng = np.random.default_rng(20260908)
    arrays = [rng.normal(size=(500, 52)).astype(np.float64) for _ in range(4)]
    transform = RobustRunTransform.fit(
        {f"run-{index}": values for index, values in enumerate(arrays)}
    )
    model = MSTCNModel.from_untrained(
        transform,
        config=backend._architecture(backend._expected_config(), 0.125),
        seed=20260908,
    )
    runs = tuple(
        backend.NumericRun(idv, 1, 50, index + 1, values)
        for index, (idv, values) in enumerate(zip((1, 3, 9, 15), arrays, strict=True))
    )
    gpu = backend._score_runs(model, runs, device="cuda", batch_size=4)
    cpu = backend._score_runs(model, runs, device="cpu", batch_size=4)
    errors = backend._parity_errors(gpu, cpu)
    assert len(errors) == 4
    assert max(errors) <= 1.0e-9
    assert all(path.endpoints.tolist() == list(range(160, 500)) for path in gpu)

