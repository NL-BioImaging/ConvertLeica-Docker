import os


block_cipher = None

PROJECT_ROOT = os.path.abspath(SPECPATH)
ENTRY_SCRIPT = os.path.join(PROJECT_ROOT, 'ConvertLeicaQT.py')
ICON_PATH = os.path.join(PROJECT_ROOT, 'favicon.ico')
SPLASH_PATH = os.path.join(PROJECT_ROOT, 'images', 'convertleica-splash.png')

# tifffile loads the LZW implementation lazily. OME-Zarr compression uses
# numcodecs.Blosc and does not require any additional imagecodecs modules.
imagecodecs_hiddenimports = ['imagecodecs._imcd']

added_datas = [
    ('ConvertLeicaQTHelp.html', '.'),
    ('favicon.ico', '.'),
    ('images', 'images'),
    ('styles', 'styles'),
]

a = Analysis(
    [ENTRY_SCRIPT],
    pathex=[PROJECT_ROOT],
    binaries=[],
    datas=added_datas,
    hiddenimports=[
        'ci_leica_converters_helpers',
        'ci_leica_converters_ometiff',
        'ci_leica_converters_ometiff_rgb',
        'ci_leica_converters_omezarr',
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
    ] + imagecodecs_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        '_tkinter',
        'PyQt5',
        # Optional integrations visible in broad development environments;
        # none are runtime dependencies of ConvertLeica.
        'pandas',
        'scipy',
        'torch',
        'pytest',
        'matplotlib',
        'PIL',
        'IPython',
        'dask',
        'distributed',
        'xarray',
        'numba',
        'llvmlite',
        'cloudpickle',
        'botocore',
        'boto3',
        's3fs',
        'zmq',
        'nbformat',
        'jedi',
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
