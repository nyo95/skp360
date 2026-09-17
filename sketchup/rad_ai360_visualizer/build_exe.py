#!/usr/bin/env python3
"""Build ai360_pipeline.exe with PyInstaller.

Run from the plugin directory:
    python build_exe.py

Output: dist/ai360_pipeline.exe  (copy to plugin dir to activate)
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent

def main():
    # Ensure PyInstaller is available
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("Installing PyInstaller...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)

    spec_content = f'''\
# -*- mode: python ; coding: utf-8 -*-
a = Analysis(
    [r"{HERE / 'ai360_worker.py'}"],
    pathex=[r"{HERE}"],
    binaries=[],
    datas=[
        (r"{HERE / 'blender_render_worker.py'}", "."),
        (r"{HERE / 'config.example.json'}", "."),
    ],
    hiddenimports=["numpy", "cv2", "urllib", "urllib.request", "urllib.error"],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="ai360_pipeline",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
'''

    spec_path = HERE / "ai360_pipeline.spec"
    spec_path.write_text(spec_content, encoding="utf-8")
    print(f"Spec: {spec_path}")

    cmd = [
        sys.executable, "-m", "PyInstaller",
        str(spec_path),
        "--distpath", str(HERE / "dist"),
        "--workpath", str(HERE / "build"),
        "--noconfirm",
        "--clean",
    ]
    print("Running PyInstaller...")
    result = subprocess.run(cmd, cwd=str(HERE))
    if result.returncode != 0:
        print("PyInstaller failed.")
        sys.exit(1)

    exe_out = HERE / "dist" / "ai360_pipeline.exe"
    if exe_out.exists():
        dest = HERE / "ai360_pipeline.exe"
        import shutil
        shutil.copy2(exe_out, dest)
        print(f"\nBuilt: {dest}  ({dest.stat().st_size // 1024} KB)")
        print("The plugin will use ai360_pipeline.exe automatically (no Python needed).")
    else:
        print(f"ERROR: exe not found at {exe_out}")
        sys.exit(1)


if __name__ == "__main__":
    main()
