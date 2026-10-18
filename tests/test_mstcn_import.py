from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def _fresh(code: str) -> subprocess.CompletedProcess[str]:
    project_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    source_root = str(project_root / "src")
    inherited = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root if not inherited else source_root + os.pathsep + inherited
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_package_import_is_completely_lazy() -> None:
    completed = _fresh(
        "import sys; "
        "assert 'numpy' not in sys.modules and 'torch' not in sys.modules; "
        "import mate_spc_mstcn; "
        "assert 'numpy' not in sys.modules and 'torch' not in sys.modules; "
        "print('lazy')"
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.strip() == "lazy"
    assert completed.stderr == ""


def test_features_first_then_model_numeric_entry_fails_closed() -> None:
    completed = _fresh(
        "import mate_spc_mstcn.features; "
        "import mate_spc_mstcn.model as m; "
        "assert m._torch is None; "
        "\ntry:\n m.build_mstcn_network()"
        "\nexcept RuntimeError as e:\n assert 'before NumPy' in str(e)"
        "\nelse:\n raise AssertionError('numeric entry did not fail closed')"
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stderr == ""


def test_fresh_model_import_preloads_torch_before_numpy() -> None:
    completed = _fresh(
        "import sys; import mate_spc_mstcn.model as m; "
        "assert m._torch is not None; "
        "assert 'torch' in sys.modules and 'numpy' in sys.modules"
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
