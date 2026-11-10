"""Build/install the Viser client fix for transparent meshes over Gaussian splats.

uv run scripts/tools/build_viser_transparency.py

Requires Node.js >=24 and npm. Re-run after reinstalling/upgrading Viser.
Only the active environment's browser bundle is replaced; Python APIs stay intact.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import viser


def main() -> None:
  client = Path(viser.__file__).resolve().parent / "client"
  source_path = Path("src/Splatting/GaussianSplats.tsx")
  source = (client / source_path).read_text()
  original = "renderOrder={10000.0 /*Generally, we want to render last.*/}"
  replacement = (
    "renderOrder={-10000.0 /*Before transparent meshes, after opaque meshes.*/}"
  )
  if source.count(original) != 1:
    raise RuntimeError(
      f"Unsupported Viser {viser.__version__}: Gaussian render stage changed. "
      "Review the client source before adapting this patch."
    )
  npm = shutil.which("npm")
  if npm is None:
    raise RuntimeError("Node.js >=24 and npm are required to build the Viser client")
  with tempfile.TemporaryDirectory(prefix="mjlab-viser-") as work:
    build_root = Path(work) / "client"
    shutil.copytree(
      client,
      build_root,
      ignore=shutil.ignore_patterns("node_modules", "build", ".git"),
    )
    (build_root / source_path).write_text(source.replace(original, replacement))
    subprocess.run(
      [npm, "ci", "--cache", str(Path(work) / "npm-cache"), "--no-audit"],
      cwd=build_root,
      check=True,
    )
    subprocess.run([npm, "run", "build"], cwd=build_root, check=True)
    built = build_root / "build/index.html"
    # Viser serves a single-file client bundle. Replace atomically after building.
    destination = client / "build/index.html"
    staged = destination.with_suffix(".mjlab.tmp")
    shutil.copyfile(built, staged)
    staged.replace(destination)
  print(f"Installed Viser {viser.__version__} transparency client: {destination}")
  print("Restart the viewer and reload the browser page to use the new bundle.")


if __name__ == "__main__":
  main()
