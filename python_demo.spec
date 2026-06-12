# -*- mode: python ; coding: utf-8 -*-

project_datas = [
    ('Resource', 'Resource'),
    ('configs', 'configs'),
    ('scripts', 'scripts'),
    ('third_party', 'third_party'),
    ('weights', 'weights'),
    ('arm_control', 'arm_control'),
    ('run_target_grasp_demo.py', '.'),
    ('run_realsense_grasp_workflow.py', '.'),
    ('qwen_config.py', '.'),
]

hiddenimports = [
    'PyQt5.QtCore',
    'PyQt5.QtGui',
    'PyQt5.QtWidgets',
    'PyQt5.QtNetwork',
    'PyQt5.QtSvg',
    'Resource.resource',
    'app.main_window',
    'app.interaction_state',
    'ipc_socket.tcp_socket_client',
    'ipc_socket.local_socket_client',
    'requests',
    'serial',
    'pyrealsense2',
    'numpy',
    'cv2',
    'PIL.Image',
    'scipy.io',
]


a = Analysis(
    ['run.py'],
    pathex=['.'],
    binaries=[],
    datas=project_datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'torchvision', 'torchaudio', 'open3d', 'groundingdino', 'segment_anything', 'transformers', 'pandas', 'sklearn', 'matplotlib'],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='python_demo',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='python_demo',
)
