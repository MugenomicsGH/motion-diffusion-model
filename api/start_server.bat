@echo off
setlocal EnableDelayedExpansion

:: Parse command line arguments
set CLEAN_INSTALL=0
if "%1"=="--clean" set CLEAN_INSTALL=1

:: Check if conda is available
where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: conda is not available in PATH
    exit /b 1
)

:: Deactivate any active conda environment
call conda deactivate 2>nul

:: Check for existing mdm environment
echo Checking for mdm environment...
conda env list | findstr /B "mdm " >nul

if %ERRORLEVEL% EQU 0 (
    if %CLEAN_INSTALL% EQU 1 (
        echo Removing existing mdm environment for clean install...
        call conda env remove -n mdm
        echo Creating fresh mdm environment...
        call conda env create -f environment_windows.yml
        if %ERRORLEVEL% NEQ 0 (
            echo Error: Failed to create conda environment
            exit /b 1
        )
    ) else (
        echo Found existing mdm environment, using it...
    )
) else (
    echo Creating mdm environment...
    call conda env create -f environment_windows.yml
    if %ERRORLEVEL% NEQ 0 (
        echo Error: Failed to create conda environment
        exit /b 1
    )
)

:: Activate the environment
echo Activating mdm environment...
call conda activate mdm
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to activate conda environment
    exit /b 1
)

:: Ensure required packages are installed
echo Ensuring required packages are installed...
pip install typing_extensions==4.5.0
pip install fastapi==0.88.0 uvicorn==0.20.0 python-multipart trimesh moviepy ftfy regex tqdm
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.7.0+cu110.html
pip install smplx==0.1.28
pip install git+https://github.com/openai/CLIP.git

:: Install spacy and download language model
echo Installing spacy and language model...
pip install spacy==3.3.1
python -m spacy download en_core_web_sm

:: Add the current directory to PYTHONPATH
set PYTHONPATH=%CD%;%PYTHONPATH%

:: Start the API server
echo Starting API server...
python -m api.api_server --host 0.0.0.0 --port 8000 --reload 