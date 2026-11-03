"""Check import-time JIT selection without weights or a CUDA compiler."""

import os
from pathlib import Path
import subprocess
import sys
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "src"
CHECK_MODE = """
import os
import torch
enabled = os.environ.get('RWKV_JIT_ON') != '0'
from rwkv import model

base = torch.jit.ScriptModule if enabled else torch.nn.Module
assert model.MyModule is base
assert model.RWKV.__bases__ == (base,)
assert isinstance(model.torch_mm8_one, torch.jit.ScriptFunction) == enabled
if os.environ['RWKV_V7_ON'] == '1':
    assert model.RWKV is model.RWKV_x070
if enabled:
    assert model.MyFunction is torch.jit.script_method
    assert model.MyStatic is torch.jit.script
else:
    def probe():
        pass
    assert model.MyFunction(probe) is probe
    assert model.MyStatic(probe) is probe
"""


class JitModeTest(unittest.TestCase):
    def check_mode(self, jit):
        for v7, deepembed in (("0", "0"), ("1", "0"), ("1", "1")):
            with self.subTest(jit=jit, v7=v7, deepembed=deepembed):
                env = os.environ.copy()
                env.update(RWKV_V7_ON=v7, RWKV_CUDA_ON="0", RWKV_DML_ON="0", RWKV_DE_VERSION=deepembed)
                env["PYTHONPATH"] = str(SOURCE)
                if jit is None:
                    env.pop("RWKV_JIT_ON", None)
                else:
                    env["RWKV_JIT_ON"] = jit
                result = subprocess.run(
                    [sys.executable, "-c", CHECK_MODE], env=env,
                    capture_output=True, text=True, timeout=60,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_jit_disabled(self):
        self.check_mode("0")

    def test_jit_enabled(self):
        self.check_mode("1")

    def test_jit_enabled_by_default(self):
        self.check_mode(None)


if __name__ == "__main__":
    unittest.main()
