<#
.SYNOPSIS
A script to set up the environment and run the transcription and diarization pipeline.

.DESCRIPTION
This script automates the following steps:
1. Checks for a Python virtual environment ('venv'). If not found, creates it.
2. Activates the virtual environment.
3. Installs required packages from requirements.txt.
4. Executes the main Python script (transcribe_diarize.py) with any provided arguments.

.PARAMETER AudioFile
(Required) The path to the audio or video file to process.

.PARAMETER Model
(Optional) The Whisper model to use (e.g., 'large-v3', 'medium.en').

.PARAMETER Language
(Optional) The language of the audio. If None, Whisper will auto-detect.

.EXAMPLE
.\run.ps1 -AudioFile "path\to\your\audio.mp4"

.EXAMPLE
.\run.ps1 -AudioFile "path\to\your\audio.wav" -Model "medium.en"
#>

[CmdletBinding()]
param (
    [Parameter(Mandatory=$true, Position=0)]
    [string]$AudioFile,

    [Parameter(Mandatory=$false)]
    [string]$Token,

    [Parameter(Mandatory=$false)]
    [string]$Model,

    [Parameter(Mandatory=$false)]
    [string]$Language,

    [Parameter(Mandatory=$false)]
    [string]$OutputDir
)

# --- 1. Check and Create Virtual Environment ---
$VenvPath = ".\venv"
if (-not (Test-Path $VenvPath)) {
    Write-Host "Virtual environment not found. Creating one at '$VenvPath'..."
    try {
        python -m venv venv
        Write-Host "Virtual environment created successfully." -ForegroundColor Green
    } catch {
        Write-Host "Error creating virtual environment. Please ensure Python 3.8+ is installed and in your PATH." -ForegroundColor Red
        exit 1
    }
}

# --- 2. Activate Virtual Environment ---
$ActivateScript = Join-Path $VenvPath "Scripts\Activate.ps1"
if (Test-Path $ActivateScript) {
    . $ActivateScript
    Write-Host "Virtual environment activated."
} else {
    Write-Host "Activation script not found at '$ActivateScript'. Cannot proceed." -ForegroundColor Red
    exit 1
}

# --- 3. Verify PyTorch and CUDA Installation ---
Write-Host "Verifying PyTorch and CUDA installation..."
$PytorchCheck = python -c "import torch; print(torch.cuda.is_available())"
if ($PytorchCheck -ne "True") {
    Write-Host "--------------------------------------------------------------------" -ForegroundColor Red
    Write-Host "[FATAL ERROR] PyTorch with CUDA support is not installed correctly." -ForegroundColor Red
    Write-Host "This script requires a manual installation of PyTorch that matches your system's CUDA version." -ForegroundColor Red
    Write-Host "Please follow the instructions in PLAN.md to install PyTorch first, then run this script again." -ForegroundColor Red
    Write-Host "--------------------------------------------------------------------" -ForegroundColor Red
    exit 1
}
Write-Host "PyTorch with CUDA support detected." -ForegroundColor Green


# --- 4. Install Other Dependencies ---
Write-Host "Checking and installing other dependencies from requirements.txt..."
try {
    pip install -r requirements.txt | Out-Null
    Write-Host "Dependencies are up to date." -ForegroundColor Green
} catch {
    Write-Host "Error installing dependencies from requirements.txt. Please check the file and your internet connection." -ForegroundColor Red
    exit 1
}


# --- 5. Run the Main Python Script ---
Write-Host "Executing the transcription and diarization pipeline..."
Write-Host "-------------------------------------------------------"

try {
    # Build the argument list for the Python script dynamically
    $pythonArgs = @($AudioFile)
    if ($PSBoundParameters.ContainsKey('Token')) { $pythonArgs += "--token", $Token }
    if ($PSBoundParameters.ContainsKey('Model')) { $pythonArgs += "--model", $Model }
    if ($PSBoundParameters.ContainsKey('Language')) { $pythonArgs += "--language", $Language }
    if ($PSBoundParameters.ContainsKey('OutputDir')) { $pythonArgs += "--output_dir", $OutputDir }

    # Use the call operator '&' to execute the python script with the constructed arguments
    & python .\transcribe_diarize.py $pythonArgs
} catch {
    Write-Host "An error occurred while running the Python script." -ForegroundColor Red
    Write-Host $_.Exception.Message
    exit 1
}

Write-Host "-------------------------------------------------------"
Write-Host "Script finished." -ForegroundColor Green