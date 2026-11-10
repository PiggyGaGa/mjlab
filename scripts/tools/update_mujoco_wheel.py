"""Update the custom release wheel without consulting uv.lock's artifact hash.

uv run --no-sync scripts/tools/update_mujoco_wheel.py
Use uv run --no-sync for subsequent commands to retain this installation.
"""

import argparse
import platform
import shlex
import subprocess
import sys


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--dry-run", action="store_true")
  args = parser.parse_args()
  if (
    sys.platform != "linux"
    or platform.machine() != "x86_64"
    or platform.python_implementation() != "CPython"
    or sys.version_info[:2] not in {(3, 12), (3, 13)}
  ):
    parser.error("The custom wheel requires Linux x86_64 and CPython 3.12/3.13")
  tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
  url = (
    "https://github.com/PiggyGaGa/mujoco/releases/download/v1.0/"
    f"mujoco-3.16.0%2B3dgs-{tag}-{tag}-manylinux_2_35_x86_64.whl"
  )
  command = [
    "uv",
    "pip",
    "install",
    "--python",
    sys.executable,
    "--reinstall",
    "--no-cache",
    "--no-deps",
    "--no-verify-hashes",
    url,
  ]
  if args.dry_run:
    command.append("--dry-run")
  print(shlex.join(command), flush=True)
  subprocess.run(command, check=True)
  print("Use uv run --no-sync to retain the installed custom wheel.")


if __name__ == "__main__":
  main()
