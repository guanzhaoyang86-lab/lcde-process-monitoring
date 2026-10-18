from __future__ import annotations

from types import SimpleNamespace
import unittest

from mate_spc.torch_backend import _tf32_disabled


class TF32ContextTests(unittest.TestCase):
    @staticmethod
    def _fake_torch(matmul: bool, cudnn: bool) -> SimpleNamespace:
        return SimpleNamespace(
            backends=SimpleNamespace(
                cuda=SimpleNamespace(
                    matmul=SimpleNamespace(allow_tf32=matmul),
                ),
                cudnn=SimpleNamespace(allow_tf32=cudnn),
            )
        )

    def test_settings_are_disabled_inside_and_restored_afterward(self) -> None:
        torch = self._fake_torch(matmul=True, cudnn=False)
        with _tf32_disabled(torch):
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertFalse(torch.backends.cudnn.allow_tf32)
        self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
        self.assertFalse(torch.backends.cudnn.allow_tf32)

    def test_settings_are_restored_after_an_exception(self) -> None:
        torch = self._fake_torch(matmul=True, cudnn=True)
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            with _tf32_disabled(torch):
                raise RuntimeError("synthetic")
        self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
        self.assertTrue(torch.backends.cudnn.allow_tf32)


if __name__ == "__main__":
    unittest.main()
