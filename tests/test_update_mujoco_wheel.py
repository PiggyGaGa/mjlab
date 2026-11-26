"""Check the supported interpreter boundary without installing a wheel."""

import importlib
from collections import namedtuple
from pathlib import Path

import pytest


@pytest.mark.parametrize("minor", [12, 13])
def test_custom_wheel_interpreter_selection(monkeypatch, minor):
  monkeypatch.syspath_prepend(
    str(Path(__file__).resolve().parents[1] / "scripts" / "tools")
  )
  module = importlib.import_module("update_mujoco_wheel")
  interpreter_version = namedtuple("InterpreterVersion", "major minor")
  monkeypatch.setattr(module.sys, "version_info", interpreter_version(3, minor))
  monkeypatch.setattr(module.sys, "platform", "linux")
  monkeypatch.setattr(module.sys, "argv", ["update_mujoco_wheel", "--dry-run"])
  monkeypatch.setattr(module.platform, "machine", lambda: "x86_64")
  monkeypatch.setattr(module.platform, "python_implementation", lambda: "CPython")
  commands = []
  monkeypatch.setattr(
    module.subprocess, "run", lambda command, **kwargs: commands.append(command)
  )
  if minor == 12:
    with pytest.raises(SystemExit) as error:
      module.main()
    assert error.value.code == 2
    assert not commands
  else:
    module.main()
    assert len(commands) == 1
    assert "--dry-run" in commands[0]
    assert any("cp313-cp313-manylinux_2_35_x86_64.whl" in arg for arg in commands[0])
