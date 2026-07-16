@echo off
setlocal
python "%~dp0omero_import_metadata_probe.py" %*
exit /b %errorlevel%

