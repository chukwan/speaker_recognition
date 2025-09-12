import os
import io
import json
from datetime import datetime
import torch
import whisper
from pyannote.audio import Pipeline
from pydub import AudioSegment
from flask import Flask, render_template, request
from flask_socketio import SocketIO
from dotenv import load_dotenv

# --- 1. Initial Setup and Model Loading ---

load_dotenv()

app = Flask(__name__)
# In a production environment, you'd want to configure a proper secret key
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# Global storage for audio buffers for each session
session_buffers = {}

# Check for CUDA device
if torch.cuda.is_available():
    DEVICE = "cuda"
    print("[INFO] CUDA is available. Using GPU for processing.")
else:
    DEVICE = "cpu"
    print("[WARNING] CUDA not available. Falling back to CPU. Processing will be significantly slower.")

# Load Whisper model
print("[INFO] Loading Whisper model...")
whisper_model = whisper.load_model("base", device=DEVICE)
print("[INFO] Whisper model loaded.")

# Load pyannote.audio pipeline
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("Hugging Face token not found. Please set HF_TOKEN in your .env file.")

print("[INFO] Loading pyannote.audio pipeline...")
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=hf_token
)
diarization_pipeline.to(torch.device(DEVICE))
print("[INFO] pyannote.audio pipeline loaded.")


# --- 2. Flask Routes and SocketIO Event Handlers ---

@app.route('/')
def index():
    """Render the main UI."""
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    """A new client has connected."""
    print(f"Client connected: {request.sid}")
    # Initialize an empty buffer for the new session
    session_buffers[request.sid] = io.BytesIO()

@socketio.on('disconnect')
def handle_disconnect():
    """A client has disconnected."""
    print(f"Client disconnected: {request.sid}")
    # Clean up the buffer for the disconnected session
    session_buffers.pop(request.sid, None)

@socketio.on('audio_chunk')
def handle_audio_chunk(chunk):
    """
    Receive an audio chunk from the client, transcribe it for a live preview,
    and append it to the session's buffer.
    """
    if request.sid in session_buffers:
        # Append the new chunk to the buffer
        buffer = session_buffers[request.sid]
        buffer.write(chunk)

        # --- Live Transcription (without diarization) ---
        # For a quick preview, we transcribe the latest chunk.
        # We need to convert it to a format Whisper understands.
        try:
            # Convert webm chunk to a processable audio segment
            audio_segment = AudioSegment.from_file(io.BytesIO(chunk), format="webm")
            audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)

            # Convert to numpy array for Whisper
            samples = torch.from_numpy(audio_segment.get_array_of_samples()).float() / 32768.0

            # Transcribe
            result = whisper_model.transcribe(samples, fp16=torch.cuda.is_available())
            text = result.get('text', '').strip()

            if text:
                socketio.emit('transcript_update', {
                    'speaker': 'Live',
                    'text': text
                })
        except Exception as e:
            print(f"[ERROR] Could not process live chunk: {e}")


def format_timestamp(seconds: float) -> str:
    """Formats a timestamp from seconds to HH:MM:SS.ms format."""
    from datetime import timedelta
    td = timedelta(seconds=seconds)
    microseconds = td.microseconds
    milliseconds = microseconds // 1000
    minutes, seconds = divmod(td.seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

@socketio.on('stop_recording')
def handle_stop_recording(data):
    """
    The client has stopped recording. Process the full audio buffer to
    perform high-quality diarization and transcription, then save the results.
    """
    session_name = data.get('session_name', 'default-session')
    print(f"[INFO] Stop command received for session: {session_name}")

    if request.sid not in session_buffers:
        print(f"[ERROR] No buffer found for session: {request.sid}")
        return

    # --- 1. Get and Convert Full Audio Buffer ---
    full_audio_buffer = session_buffers[request.sid]
    full_audio_buffer.seek(0)

    try:
        audio = AudioSegment.from_file(full_audio_buffer, format="webm")
        audio = audio.set_frame_rate(16000).set_channels(1)
    except Exception as e:
        print(f"[ERROR] Failed to convert full audio buffer: {e}")
        return

    # --- 2. Create Session Directory ---
    output_dir = os.path.join("sessions", session_name)
    os.makedirs(output_dir, exist_ok=True)

    # --- 3. Save Audio to MP3 ---
    mp3_path = os.path.join(output_dir, f"{session_name}.mp3")
    print(f"[INFO] Saving audio to {mp3_path}")
    audio.export(mp3_path, format="mp3")

    # --- 4. Perform High-Quality Diarization ---
    print("[INFO] Performing final diarization...")
    # Convert audio to numpy array for models
    audio_waveform = torch.from_numpy(audio.get_array_of_samples()).float() / 32768.0
    diarization_input = {
        "waveform": audio_waveform.unsqueeze(0),
        "sample_rate": 16000
    }
    diarization = diarization_pipeline(diarization_input)
    print("[SUCCESS] Final diarization complete.")

    # --- 5. Transcribe Segments and Create Markdown ---
    print("[INFO] Transcribing final segments...")
    final_transcript = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segment_start = turn.start * 1000  # pydub works in milliseconds
        segment_end = turn.end * 1000
        segment_audio = audio[segment_start:segment_end]

        segment_samples = torch.from_numpy(segment_audio.get_array_of_samples()).float() / 32768.0

        result = whisper_model.transcribe(segment_samples, fp16=torch.cuda.is_available())
        text = result.get('text', '').strip()

        if text:
            final_transcript.append({
                "start": turn.start,
                "speaker": speaker,
                "text": text
            })

    # Sort by start time
    final_transcript.sort(key=lambda x: x['start'])

    md_path = os.path.join(output_dir, f"{session_name}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Diarization for {session_name}\n\n")
        for item in final_transcript:
            start_time = format_timestamp(item['start'])
            f.write(f"**{item['speaker']}** ({start_time}): {item['text']}\n\n")
    print(f"[SUCCESS] Transcript saved to {md_path}")

    # --- 6. Update JSON Database ---
    db_path = "data/tasks.json"
    tasks = []
    if os.path.exists(db_path):
        with open(db_path, "r") as f:
            tasks = json.load(f)

    tasks.append({
        "filename": session_name,
        "created_date": datetime.now().isoformat(),
        "saved_location_path_of_the_audio": mp3_path,
        "saved_location_path_of_the_transcript": md_path,
    })

    with open(db_path, "w") as f:
        json.dump(tasks, f, indent=4)

    # --- 7. Notify Client ---
    socketio.emit('processing_complete', {'session_name': session_name})
    print(f"[SUCCESS] Session '{session_name}' processed and saved.")


if __name__ == '__main__':
    print("[INFO] Starting Flask-SocketIO server.")
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)
