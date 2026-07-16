import os

from PyInstaller.utils.hooks import collect_all


block_cipher = None

PROJECT_ROOT = os.path.abspath(SPECPATH)
ENTRY_SCRIPT = os.path.join(PROJECT_ROOT, 'ConvertLeicaQT.py')
ICON_PATH = os.path.join(PROJECT_ROOT, 'favicon.ico')
SPLASH_PATH = os.path.join(PROJECT_ROOT, 'images', 'convertleica-splash.png')

pyqt6_datas, pyqt6_binaries, pyqt6_hiddenimports = collect_all('PyQt6')
cv2_datas, cv2_binaries, cv2_hiddenimports = collect_all('cv2')
numpy_datas, numpy_binaries, numpy_hiddenimports = collect_all('numpy')
tifffile_datas, tifffile_binaries, tifffile_hiddenimports = collect_all('tifffile')
imagecodecs_datas, imagecodecs_binaries, imagecodecs_hiddenimports = collect_all('imagecodecs')

added_datas = [
    ('ConvertLeicaQTHelp.html', '.'),
    ('favicon.ico', '.'),
    ('images', 'images'),
    ('styles', 'styles'),
]

a = Analysis(
    [ENTRY_SCRIPT],
    pathex=[PROJECT_ROOT],
    binaries=pyqt6_binaries + cv2_binaries + numpy_binaries + tifffile_binaries + imagecodecs_binaries,
    datas=added_datas + pyqt6_datas + cv2_datas + numpy_datas + tifffile_datas + imagecodecs_datas,
    hiddenimports=[
        'ci_leica_converters_helpers',
        'ci_leica_converters_ometiff',
        'ci_leica_converters_ometiff_rgb',
        'ci_leica_converters_single_lif',
        'cideconvolve_io.ome_tiff_io',
        'leica_converter',
        'CreatePreview',
        'ParseLeicaImageXML',
        'ParseLeicaImageXMLLite',
        'ReadLeicaLIF',
        'ReadLeicaLOF',
        'ReadLeicaXLEF',
        'encodings.idna',
    ] + pyqt6_hiddenimports + cv2_hiddenimports + numpy_hiddenimports + tifffile_hiddenimports + imagecodecs_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        '_tkinter',
        'PyQt5',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

splash = Splash(
    SPLASH_PATH,
    binaries=a.binaries,
    datas=a.datas,
    text_pos=(20, 356),
    text_size=10,
    text_color='#dce7ef',
    text_default='Loading ConvertLeica...',
    always_on_top=True,
)

exe = EXE(
    pyz,
    a.scripts,
    splash,
    splash.binaries,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='ConvertLeicaGUI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon=ICON_PATH,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
