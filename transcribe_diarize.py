import torch
import whisper
import torchaudio
from pyannote.audio import Pipeline
from huggingface_hub.errors import HfHubHTTPError
import os
import argparse
from datetime import timedelta
import subprocess
import tempfile
from dotenv import load_dotenv
from tqdm import tqdm

def format_timestamp(seconds: float) -> str:
    """
    Formats a timestamp from seconds to HH:MM:SS.ms format.
    """
    td = timedelta(seconds=seconds)
    microseconds = td.microseconds
    milliseconds = microseconds // 1000
    minutes, seconds = divmod(td.seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

def convert_to_wav(input_file: str) -> str:
    """
    Converts an audio/video file to a temporary WAV file using ffmpeg.
    Returns the path to the temporary WAV file.
    """
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_wav_path = temp_wav.name
    temp_wav.close()

    print(f"\n[INFO] Converting '{input_file}' to temporary WAV file...")
    print("[INFO] ffmpeg output:")
    
    command = [
        "ffmpeg",
        "-i", input_file,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        "-y", # Overwrite output file if it exists
        temp_wav_path
    ]

    try:
        # Use Popen to stream ffmpeg output in real-time
        process = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding='utf-8')

        # Print ffmpeg's stderr line by line (it outputs progress here)
        while True:
            line = process.stderr.readline()
            if not line:
                break
            print(line.strip())

        process.wait()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command, output=process.stdout.read(), stderr=process.stderr.read())

        print("\n[SUCCESS] Conversion successful.")
        return temp_wav_path
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Error during ffmpeg conversion: {e.stderr}")
        # Clean up the empty temp file if conversion fails
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
        raise
    except FileNotFoundError:
        print("Error: ffmpeg is not installed or not in your system's PATH.")
        print("Please install ffmpeg to process non-WAV files.")
        # Clean up the empty temp file
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
        raise

def transcribe_diarize(audio_path: str, hf_token: str | None, model_name: str = "large-v3", language: str | None = None, output_dir: str = "transcripts"):
    """
    Performs speaker diarization and transcription on an audio file.

    Args:
        audio_path (str): Path to the local audio file.
        hf_token (str | None): Your Hugging Face access token. If None, it's loaded from .env.
        model_name (str): The Whisper model to use (e.g., "large-v3", "medium.en").
        language (str | None): The language of the audio. If None, Whisper will auto-detect.
        output_dir (str): The directory to save the output transcript file.
    """
    # 1. Setup: Determine device and load models
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at '{audio_path}'")
        return

    if hf_token is None:
        print("Error: Hugging Face token is not provided. Please pass it via --token or set it in a .env file.")
        return

    # --- Device Selection ---
    # --- Device Selection ---
    print("\n[DEBUG] Checking for available processing device...")
    if torch.cuda.is_available():
        device = "cuda"
        gpu_props = torch.cuda.get_device_properties(0)
        print(f"[SUCCESS] CUDA-enabled GPU detected: {gpu_props.name}")
        print(f"          Total Memory: {gpu_props.total_memory / 1e9:.2f} GB")
        print("          Reason: torch.cuda.is_available() returned True. Pipeline will use GPU.")
        if "large" in model_name and gpu_props.total_memory < 10e9:
             print("\n[WARNING] The 'large' Whisper model is VRAM-intensive. Consider a smaller model if you encounter memory issues.")
    else:
        print("\n[FATAL ERROR] No CUDA-enabled GPU found.")
        print("              Reason: torch.cuda.is_available() returned False.")
        print("              This pipeline is optimized for GPU and will not run in CPU-only mode.")
        print("\n              TROUBLESHOOTING:")
        print("              1. Ensure you have a compatible NVIDIA GPU and the latest drivers installed.")
        print("              2. The `requirements.txt` file is configured to install the correct PyTorch version for CUDA 12.1.")
        print("              3. Re-run the setup to ensure the correct dependencies are installed:")
        print("                 - In PowerShell, run: .\\run.ps1 YOUR_AUDIO_FILE.mp4")
        print("                 - Or manually: pip install -r requirements.txt --force-reinstall")
        return # Exit the function

    print(f"\n[INFO] Loading Whisper model: '{model_name}'... (This may take a moment)")
    whisper_model = whisper.load_model(model_name, device=device)

    print("[INFO] Loading speaker diarization pipeline... (This may take a moment)")
    try:
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        diarization_pipeline.to(torch.device(device))
    except HfHubHTTPError as e:
        print("\n[ERROR] Failed to load pyannote.audio pipeline.")
        print("This is likely due to an invalid Hugging Face token or not accepting the model's user agreement.")
        print("\nPlease verify the following:")
        print("1. Your HF_TOKEN in the .env file is correct.")
        print("2. You have accepted the user agreement for the models on Hugging Face Hub:")
        print("   - Diarization Model: https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("   - Segmentation Model: https://huggingface.co/pyannote/segmentation-3.0")
        print(f"\nOriginal Error: {e}")
        return
    print("[SUCCESS] Models loaded successfully.")

    # 2. Pre-process Audio: Convert to WAV if necessary
    temp_wav_file = None
    if not audio_path.lower().endswith('.wav'):
        try:
            temp_wav_file = convert_to_wav(audio_path)
            processing_path = temp_wav_file
        except (FileNotFoundError, subprocess.CalledProcessError):
            return # Stop execution if conversion fails
    else:
        processing_path = audio_path

    # 3. Load Audio
    print(f"[INFO] Loading audio from: {processing_path}")
    audio_waveform = whisper.load_audio(processing_path)

    # 4. Speaker Diarization
    print("[INFO] Performing speaker diarization... (This can take a while for long audio)")
    # Note: pyannote.audio's diarization pipeline involves CPU-intensive clustering
    # after the initial GPU-based neural network inference. High CPU usage here is expected.
    diarization = diarization_pipeline(
        {"waveform": torch.from_numpy(audio_waveform).unsqueeze(0), "sample_rate": 16000},
        min_speakers=2,
        max_speakers=5 # Adjust as needed
    )
    print("[SUCCESS] Diarization complete.")

    # 5. Transcribe per speaker segment
    print("\n[INFO] Transcribing segments...")
    final_transcript = []
    
    # Use tqdm for a progress bar
    diarization_list = list(diarization.itertracks(yield_label=True))
    for turn, _, speaker in tqdm(diarization_list, desc="Transcription Progress"):
        start_time = turn.start
        end_time = turn.end
        
        # Extract the audio segment for this turn
        segment_waveform = audio_waveform[int(start_time * 16000):int(end_time * 16000)]
        
        # Convert numpy array to torch tensor
        segment_tensor = torch.from_numpy(segment_waveform).to(device)

        # Transcribe the segment
        # Use fp16=True for significant speedup on compatible GPUs
        # Perform transcription
        transcribe_options = {
            "fp16": torch.cuda.is_available()
        }
        if language:
            transcribe_options["language"] = language
        
        result = whisper_model.transcribe(segment_tensor, **transcribe_options)
        
        if result["text"].strip(): # Only add if there is transcribed text
            final_transcript.append({
                "start": start_time,
                "end": end_time,
                "speaker": speaker,
                "text": result["text"].strip()
            })

    # 5. Sort and Print Final Transcript
    print("[SUCCESS] Transcription finished.")

    # 6. Sort and Print Final Transcript
    print("\n--- FINAL TRANSCRIPT ---")
    # Sort by start time to ensure chronological order
    final_transcript.sort(key=lambda x: x['start'])
    
    output_lines = []
    for item in final_transcript:
        start_str = format_timestamp(item['start'])
        end_str = format_timestamp(item['end'])
        line = f"[{start_str} --> {end_str}] {item['speaker']}: {item['text']}"
        print(line)
        output_lines.append(line)

    # 7. Save to file
    output_filename = os.path.splitext(os.path.basename(audio_path))[0] + ".txt"
    output_filepath = os.path.join(output_dir, output_filename)
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
        
    print(f"\n[SUCCESS] Transcript saved to: {output_filepath}")

    # 8. Cleanup
    if temp_wav_file:
        print(f"\n[INFO] Removing temporary file: {temp_wav_file}")
        os.remove(temp_wav_file)


if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()

    parser = argparse.ArgumentParser(description="Offline Transcription and Speaker Diarization Pipeline")
    parser.add_argument("audio_file", type=str, help="Path to the audio file to process.")
    parser.add_argument(
        "--token",
        type=str,
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face authentication token. If not provided, it will be read from the HF_TOKEN environment variable."
    )
    parser.add_argument("--model", type=str, default="large-v3", help="Whisper model name (e.g., 'tiny', 'base', 'medium.en', 'large-v3').")
    parser.add_argument("--language", type=str, default=None, help="Language of the audio for transcription (e.g., 'en', 'zh', 'fr'). If omitted, Whisper will auto-detect.")
    parser.add_argument("--output_dir", type=str, default="transcripts", help="Directory to save the output transcript file.")
    
    args = parser.parse_args()

    transcribe_diarize(args.audio_file, args.token, args.model, args.language, args.output_dir)