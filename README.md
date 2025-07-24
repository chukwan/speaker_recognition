# High-Performance Offline Transcription and Speaker Diarization Pipeline

This project provides a complete, high-performance pipeline for offline audio transcription and speaker diarization. It is optimized for NVIDIA GPUs and uses state-of-the-art local models to answer the question: "who spoke what, and when?"

The pipeline integrates two core components:
1.  **Transcription:** OpenAI's Whisper (`large-v3` model by default) for highly accurate speech-to-text.
2.  **Speaker Diarization:** `pyannote.audio` to identify and segment different speakers.

---

### **Step 1: Environment Setup & Prerequisites**

This is the foundational step to ensure all software and hardware components are ready.

**1.1. Hardware Requirements:**
- **NVIDIA GPU:** An RTX 30-series or 40-series GPU (e.g., RTX 4090, 3090) with at least 10GB of VRAM is strongly recommended for running the `large-v3` Whisper model efficiently.
- **System RAM:** 16GB or more.

**1.2. Software & Driver Installation:**
- **FFmpeg:** Install the FFmpeg command-line tool. This is essential for converting various audio and video formats into a compatible WAV format for processing. You can download it from [ffmpeg.org](https://ffmpeg.org/download.html).
- **NVIDIA CUDA Toolkit:** Install a recent version of the CUDA Toolkit that is compatible with PyTorch. You can verify your driver version by running `nvidia-smi` in the terminal.
- **Python:** Ensure you have Python 3.8+ installed.

**1.3. Install Python Libraries (2-Step Process):**

**Step 1: Install PyTorch with CUDA Support (Manual)**
This is the most critical step. You must install the version of PyTorch that exactly matches your system's CUDA toolkit.
1.  Go to the official PyTorch website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
2.  Use the interactive tool to select your system configuration (e.g., Stable, Windows, Pip, Python, CUDA 12.x).
3.  Copy the generated command and run it in your activated virtual environment. It will look something like this:
    ```powershell
    # Example for a specific CUDA version - use the one from the website!
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

**Step 2: Install Remaining Dependencies**
After PyTorch is installed correctly, you can install the rest of the required libraries.
- The `run.ps1` script will do this for you automatically after it verifies your PyTorch installation.
- To do it manually, run:
  ```bash
  pip install -r requirements.txt
  ```

**1.4. Hugging Face Authentication:**

> **Why is this needed?** The `pyannote.audio` library requires authentication to download its pre-trained speaker diarization models. The model creators have "gated" them, meaning you must agree to their terms of use on the Hugging Face website before access is granted. The token simply proves you have done so.

- **Accept User Agreements:** Visit the following pages and accept the user conditions:
    1.  [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1)
    2.  [`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0)
- **Generate Access Token:** Create a Hugging Face access token from your account: **Settings > Access Tokens > New token**. The token only requires **read** permissions.
- **Create `.env` file:** For securely storing your token, create a file named `.env` in the root of the project. You can copy the provided `.env.example` file:
  ```bash
  cp .env.example .env
  ```
  Then, open the `.env` file and replace the placeholder with your actual Hugging Face token.

---

### **Step 2: Execute the Pipeline**

This is the final step to run the script and get the output.

**2.1. Using the PowerShell Script (Recommended for Windows):**
- Open a PowerShell terminal.
- You may need to adjust your execution policy to run local scripts. You can do this for the current session by running:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
  ```
- Execute the `run.ps1` script, passing the path to your audio/video file as an argument. The script handles virtual environment creation, activation, and dependency installation automatically.
  ```powershell
  .\run.ps1 "path\to\your\audio.mp4"
  ```
- To pass additional arguments to the Python script (like changing the model, specifying the language, or setting an output directory), append them after the file path:
  ```powershell
  .\run.ps1 "path\to\your\audio.mp4" --model "medium.en" --language "zh" --output_dir "C:\custom_folder"
  ```

**2.2. Manual Execution:**
- Open your terminal and activate the virtual environment.
  ```bash
  # On Windows
  .\venv\Scripts\activate
  # On macOS/Linux
  source venv/bin/activate
  ```
- Execute the Python script directly:
  ```bash
  python transcribe_diarize.py "path/to/your/audio.wav" --model "large-v3" --language "zh" --output_dir "C:\custom_folder"
  ```
- The script will download models on the first run, then process the audio, print the final transcript, and save it to a `.txt` file inside the `transcripts/` folder (or a custom one if specified). If the `--language` flag is omitted, Whisper will attempt to auto-detect the language.

---

### **Step 3: Troubleshooting**

**Error: `No CUDA-enabled GPU found` or `PyTorch with CUDA support is not installed correctly`**

This error means `torch.cuda.is_available()` is returning `False`. The script is configured to exit if this happens. This is almost always an environment issue where the installed PyTorch library does not match your system's NVIDIA drivers or a CPU-only version was installed.

**Action:** You must manually install the correct PyTorch version.
1.  Delete your `venv` folder to start fresh.
2.  Create and activate a new virtual environment.
3.  Follow the instructions in **Step 1.3** above to install the correct PyTorch version from their official website.
4.  Once PyTorch is installed correctly, you can run the `run.ps1` script, which will then install the other dependencies and execute the pipeline.

**Error: `401 Client Error: Unauthorized`**

If you see a `401 Unauthorized` error when the script tries to load the `pyannote` pipeline, it means there is a problem with your Hugging Face authentication. The script has been updated to provide a detailed error message, but here are the steps to fix it:

1.  **Check your `.env` file:** Ensure the `HF_TOKEN` in your `.env` file is exactly correct, with no extra spaces or characters.
2.  **Accept the User Agreements:** This is the most common cause. You *must* be logged into your Hugging Face account and visit the two model pages below to accept their terms.
    - [Accept for Diarization Model](https://huggingface.co/pyannote/speaker-diarization-3.1)
    - [Accept for Segmentation Model](https://huggingface.co/pyannote/segmentation-3.0)
3.  **Check your Token Permissions:** Ensure the token you generated has at least "read" permissions.

**Observation: High CPU Usage During Diarization**

You are correct to notice high CPU usage when the script is "Performing speaker diarization...". This is expected behavior.

- **Whisper (Transcription):** Runs almost entirely on the GPU.
- **Pyannote (Diarization):** This is a multi-step process.
    1.  **Neural Inference:** The core parts (speech activity detection, speaker embedding) are sent to the GPU via the `.to(device)` command and will utilize CUDA.
    2.  **Clustering:** After the neural networks extract speaker information, a clustering algorithm runs to group the segments by speaker. This clustering part of the pipeline is highly CPU-intensive and is designed to run on the CPU.

So, while the script correctly offloads the neural network calculations to the GPU, the subsequent processing by `pyannote` will still cause a significant, but temporary, spike in CPU load.

---

### **Step 4: Configuring VS Code (Optional)**

**Issue: `Import "torch" could not be resolved` in the editor.**

If you see errors like this in your VS Code editor, it means the editor is not using the Python interpreter from your virtual environment (`venv`). This is an IDE configuration issue and **will not prevent the script from running** via the `run.ps1` terminal script.

However, to fix the error highlighting and get proper code completion (IntelliSense), you need to tell VS Code to use the virtual environment's Python interpreter.

**Action:**
1.  Open the Command Palette: `Ctrl+Shift+P`
2.  Type and select **Python: Select Interpreter**.
3.  A list of available interpreters will appear. Choose the one that includes `.\venv\Scripts\python.exe`.
4.  Once selected, VS Code will use the correct interpreter, and the Pylance import errors should disappear.