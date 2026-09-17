# -*- mode: python ; coding: utf-8 -*-
a = Analysis(
    [r"C:\Users\berka\AppData\Roaming\SketchUp\SketchUp 2021\SketchUp\Plugins\rad_ai360_visualizer\ai360_worker.py"],
    pathex=[r"C:\Users\berka\AppData\Roaming\SketchUp\SketchUp 2021\SketchUp\Plugins\rad_ai360_visualizer"],
    binaries=[],
    datas=[
        (r"C:\Users\berka\AppData\Roaming\SketchUp\SketchUp 2021\SketchUp\Plugins\rad_ai360_visualizer\blender_render_worker.py", "."),
        (r"C:\Users\berka\AppData\Roaming\SketchUp\SketchUp 2021\SketchUp\Plugins\rad_ai360_visualizer\config.example.json", "."),
    ],
    hiddenimports=["numpy", "cv2", "urllib", "urllib.request", "urllib.error"],
    hookspath=[],
    hooksconfig={},
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
