import os
os.environ["NUMBA_DISABLE_JIT"] = "0"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "hf_cache")
TEMP_DIR = os.path.join(BASE_DIR, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)
os.environ["TEMP"] = TEMP_DIR
os.environ["TMP"] = TEMP_DIR
import tempfile
tempfile.tempdir = TEMP_DIR

import gradio as gr
import subprocess
import sys
import shutil
import re
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

SAVED_VOICES_DIR = os.path.join(BASE_DIR, "saved_voices")
os.makedirs(SAVED_VOICES_DIR, exist_ok=True)


def find_executable(path, exe_names):
    if os.path.isfile(path):
        return path
    if isinstance(exe_names, str):
        exe_names = [exe_names]
    for name in exe_names:
        exe_path = shutil.which(name)
        if exe_path:
            return exe_path
    return None

import sys

# Platform-aware edge-tts path resolution to prevent executing Windows binary in Linux containers
if sys.platform == "win32":
    edge_tts_default = os.path.join(BASE_DIR, "venv", "Scripts", "edge-tts.exe")
else:
    edge_tts_default = os.path.join(BASE_DIR, "venv", "bin", "edge-tts")

EDGE_TTS_EXE = find_executable(
    edge_tts_default,
    ["edge-tts", "edge-tts.exe"]
)

# Search paths for F5-TTS Python interpreter, including host venvs and relocated container venvs
F5_TTS_PYTHON_EXECUTABLE_PATHS = [
    os.path.join(BASE_DIR, "venvs", "f5tts", "Scripts", "python.exe"),
    os.path.join(BASE_DIR, "venvs", "f5tts", "bin", "python"),
    "/venvs/f5tts/bin/python",
]
F5_TTS_PYTHON_EXE = find_executable(
    F5_TTS_PYTHON_EXECUTABLE_PATHS[0],
    [F5_TTS_PYTHON_EXECUTABLE_PATHS[0], F5_TTS_PYTHON_EXECUTABLE_PATHS[1], F5_TTS_PYTHON_EXECUTABLE_PATHS[2], "python"]
)

RVC_MODELS_DIR = os.path.join(BASE_DIR, "rvc_models")

# Search paths for RVC python environment, including host venvs and relocated container venvs
RVC_PYTHON_EXECUTABLE_PATHS = [
    os.path.join(BASE_DIR, "venvs", "rvc", "Scripts", "python.exe"),
    os.path.join(BASE_DIR, "venvs", "rvc", "bin", "python"),
    "/venvs/rvc/bin/python",
]
RVC_PYTHON_EXE = find_executable(
    RVC_PYTHON_EXECUTABLE_PATHS[0],
    [RVC_PYTHON_EXECUTABLE_PATHS[0], RVC_PYTHON_EXECUTABLE_PATHS[1], RVC_PYTHON_EXECUTABLE_PATHS[2], "python"]
)

RVC_INFER_SCRIPT = os.path.join(BASE_DIR, "rvc_infer.py")
os.makedirs(RVC_MODELS_DIR, exist_ok=True)

# ─── Utility: Run edge-tts via subprocess (avoids asyncio conflicts with Gradio) ───
def run_edge_tts(text, voice, output_path, rate=None, pitch=None):
    """Generate TTS audio using edge-tts CLI. Returns True on success."""
    if not EDGE_TTS_EXE:
        return False, "edge-tts executable not found. Make sure it is installed or available in PATH."
    cmd = [EDGE_TTS_EXE, "--voice", voice, "--text", text, "--write-media", output_path]
    if rate:
        cmd += ["--rate", rate]
    if pitch:
        cmd += ["--pitch", pitch]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    return result.returncode == 0, result.stderr or result.stdout


def normalize_character_name(name):
    if not name:
        return ""
    return re.sub(r"\s+", " ", name.strip())


def normalize_saved_voice_name(name):
    if not name:
        return ""
    return name.strip().replace(" ", "_")


def match_saved_voice_name(name, saved_voices):
    if not name:
        return None
    target = normalize_character_name(name).lower()
    for saved in saved_voices:
        if saved.lower() == target or saved.lower().replace("_", " ") == target:
            return saved
    return None


def contains_devanagari_or_arabic(text):
    return any(
        '\u0900' <= c <= '\u097F' or
        '\u0600' <= c <= '\u06FF' or
        '\u0750' <= c <= '\u077F'
        for c in text
    )


def is_likely_roman_hindi_or_urdu(text):
    words = set(re.findall(r"[A-Za-z]+", text.lower()))
    common = {
        'kya', 'hai', 'hain', 'ka', 'ki', 'ke', 'ko', 'se', 'me', 'mein', 'aur', 'ya',
        'nahi', 'hum', 'tum', 'aap', 'wo', 'ye', 'woh', 'yeh', 'tha', 'thi', 'the',
        'hoga', 'hogi', 'haal', 'bhai', 'yaar', 'dost', 'kaise', 'kaisa', 'bahut',
        'accha', 'bura', 'theek', 'sab', 'lekin', 'magar', 'kyunki', 'abhi', 'kahani',
        'karo', 'hain', 'hain', 'hai'
    }
    return len(words & common) >= 2


def is_hindi_urdu_text(text):
    if not text or not text.strip():
        return False
    return contains_devanagari_or_arabic(text) or is_likely_roman_hindi_or_urdu(text)


def detect_script_language(text):
    """Detect if text is written in Arabic script (Urdu) or otherwise (Hindi/Roman)."""
    has_arabic = any('\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F' for c in text)
    if has_arabic:
        return "urdu"
    return "hindi"


def guess_voice_gender(voice_name):
    """Heuristic to guess character voice gender to align pitch of the base TTS voice."""
    if not voice_name:
        return "male"
    name_lower = voice_name.lower()
    female_indicators = [
        "female", "girl", "woman", "lady", "she", "her", "aria", "swara", 
        "uzma", "gul", "sakura", "hinata", "nami", "robin", "mikasa", 
        "asuka", "rei", "lucy", "erza", "tsunade", "kaguya", "nezuko", 
        "miku", "chika", "rem", "ram", "emilia", "alice", "samantha"
    ]
    if any(indicator in name_lower for indicator in female_indicators):
        return "female"
    return "male"


def generate_native_pronunciation_audio(text, voice_name, progress=gr.Progress()):
    if not text:
        return None, "No text provided for pronunciation generation."
    
    # Detect language script and character voice gender
    lang = detect_script_language(text)
    gender = guess_voice_gender(voice_name)
    
    # Select the optimal Microsoft Neural base voice ID
    if lang == "urdu":
        voice_id = "ur-PK-UzmaNeural" if gender == "female" else "ur-PK-AsadNeural"
    else:
        # For Hindi, hi-IN-SwaraNeural is female, hi-IN-MadhurNeural is male
        voice_id = "hi-IN-SwaraNeural" if gender == "female" else "hi-IN-MadhurNeural"
        
    final_text = text
    # Only transliterate to Devanagari if it is a Hindi/Roman script and doesn't already contain Devanagari
    if lang == "hindi" and not any('\u0900' <= c <= '\u097F' for c in text):
        try:
            from transliterate import roman_to_devanagari
            final_text = roman_to_devanagari(text)
        except Exception:
            final_text = text

    output_path = os.path.join(TEMP_DIR, f"native_pronunciation_{voice_name}.wav")
    ok, err = run_edge_tts(final_text, voice_id, output_path)
    if not ok:
        return None, err
    return output_path, final_text


# ─── Voice Library ───
def get_saved_voices():
    voices = []
    if os.path.exists(SAVED_VOICES_DIR):
        for d in sorted(os.listdir(SAVED_VOICES_DIR)):
            if os.path.isdir(os.path.join(SAVED_VOICES_DIR, d)):
                voices.append(d)
    return voices

def load_voice(name):
    if not name:
        return None, ""
    audio_path = os.path.join(SAVED_VOICES_DIR, name, "audio.wav")
    text_path = os.path.join(SAVED_VOICES_DIR, name, "text.txt")
    text = ""
    if os.path.exists(text_path):
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()
    if not os.path.exists(audio_path):
        return None, text
    return audio_path, text

def save_voice(name, audio_path, text):
    if not name or not audio_path:
        return "❌ Provide a name AND audio file.", gr.update(), gr.update(), gr.update(), gr.update()
    name = name.strip().replace(" ", "_")
    voice_dir = os.path.join(SAVED_VOICES_DIR, name)
    os.makedirs(voice_dir, exist_ok=True)
    shutil.copy(audio_path, os.path.join(voice_dir, "audio.wav"))
    with open(os.path.join(voice_dir, "text.txt"), "w", encoding="utf-8") as f:
        f.write(text or "")
    choices = get_saved_voices()
    return f"✅ Voice '{name}' saved!", gr.update(choices=choices, value=name), gr.update(choices=choices), gr.update(choices=choices), gr.update(choices=choices)

def delete_voice(name):
    if not name:
        return "Select a voice first.", gr.update(), gr.update(), gr.update(), gr.update()
    voice_dir = os.path.join(SAVED_VOICES_DIR, name)
    if os.path.exists(voice_dir):
        shutil.rmtree(voice_dir)
    choices = get_saved_voices()
    return f"🗑️ Deleted '{name}'", gr.update(choices=choices, value=None), gr.update(choices=choices), gr.update(choices=choices), gr.update(choices=choices)

def refresh_library():
    choices = get_saved_voices()
    return gr.update(choices=choices), gr.update(choices=choices), gr.update(choices=choices), gr.update(choices=choices)

# ─── RVC Backend ───
def get_rvc_models():
    models = []
    if os.path.exists(RVC_MODELS_DIR):
        for f in os.listdir(RVC_MODELS_DIR):
            if f.endswith(".pth"):
                models.append(f)
    return models

def run_rvc_conversion(input_audio, model_name, pitch):
    if not input_audio:
        return None, "Please upload a reference audio."
    if not model_name:
        return None, "Please select an RVC model (.pth)."
    if not RVC_PYTHON_EXE:
        return None, "RVC Python executable not found. Check your installation or PATH."
    model_path = os.path.join(RVC_MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        return None, f"RVC model not found: {model_name}"
    output_path = os.path.join(BASE_DIR, "rvc_output.wav")
    
    cmd = [
        RVC_PYTHON_EXE, RVC_INFER_SCRIPT,
        "--model", model_path,
        "--input", input_audio,
        "--output", output_path,
        "--pitch", str(int(pitch)),
        "--method", "rmvpe"
    ]
    
    # Try finding an index file with the same name
    index_path = model_path.replace(".pth", ".index")
    if os.path.exists(index_path):
        cmd += ["--index", index_path]
        
    env = os.environ.copy()
    
    # CPU Threading optimizations to prevent thread contention and speed up inference on CPU
    import multiprocessing
    cores = str(max(1, multiprocessing.cpu_count() // 2))
    env.update({
        "OMP_NUM_THREADS": cores,
        "MKL_NUM_THREADS": cores,
        "OPENBLAS_NUM_THREADS": cores,
        "NUMEXPR_NUM_THREADS": cores,
        "VECLIB_MAXIMUM_THREADS": cores
    })
        
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', env=env)
    if result.returncode == 0 and os.path.exists(output_path):
        return output_path, "✅ Voice converted successfully!"
    return None, f"❌ RVC Error:\n{result.stdout}\n{result.stderr}"


# ─── F5-TTS Core Engine ───
def run_f5tts(text, ref_audio_path, ref_text, output_name="output_cloned.wav", progress=None, progress_base=0.4, progress_scale=0.55, nfe_step=10):
    if not F5_TTS_PYTHON_EXE:
        return None, "F5-TTS Python executable not found. Check your installation or PATH."
    output_path = os.path.join(BASE_DIR, output_name)
    trimmed = os.path.join(TEMP_DIR, "trimmed_ref_gen.wav")

    audio = AudioSegment.from_file(ref_audio_path)
    if len(audio) > 8000:
        audio = audio[:8000]
    audio.export(trimmed, format="wav")

    if os.path.exists(output_path):
        os.remove(output_path)
        
    # Prevent internal F5-TTS whisper from hanging on low VRAM
    if not ref_text or not ref_text.strip():
        class DummyProgress:
            def __call__(self, *args, **kwargs):
                pass
        ref_text = extract_text_fn(trimmed, progress=DummyProgress())
        if ref_text.startswith("Error"):
            return None, f"Failed to transcribe reference audio: {ref_text}"

    import tomli_w
    config_path = os.path.join(BASE_DIR, "inference_config.toml")
    config_dict = {
        "model": "F5TTS_Base", "ref_audio": trimmed,
        "ref_text": ref_text.strip(),
        "speed": 1.0, "nfe_step": int(nfe_step), "gen_text": text,
        "output_dir": BASE_DIR, "output_file": output_name, "voices": {}
    }
    with open(config_path, "wb") as f:
        tomli_w.dump(config_dict, f)

    env = os.environ.copy()
    
    # CPU Threading optimizations to prevent thread contention and speed up inference on CPU
    import multiprocessing
    import torch
    cores = str(max(1, multiprocessing.cpu_count() // 2))
    env.update({
        "TEMP": TEMP_DIR,
        "TMP": TEMP_DIR,
        "NUMBA_DISABLE_JIT": "0",
        "HF_HOME": os.environ["HF_HOME"],
        "PYTHONIOENCODING": "utf-8",
        "OMP_NUM_THREADS": cores,
        "MKL_NUM_THREADS": cores,
        "OPENBLAS_NUM_THREADS": cores,
        "NUMEXPR_NUM_THREADS": cores,
        "VECLIB_MAXIMUM_THREADS": cores
    })
    
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    process = subprocess.Popen(
        [F5_TTS_PYTHON_EXE, "-m", "f5_tts.infer.infer_cli", "-c", config_path, "-o", BASE_DIR, "-w", output_name, "--device", device_str],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        env=env,
        bufsize=1
    )

    stdout_lines = []
    buffer = ""
    while True:
        char = process.stdout.read(1)
        if not char:
            break
        if char in ('\r', '\n'):
            line = buffer.strip()
            buffer = ""
            if line:
                stdout_lines.append(line)
                # Parse progress percentage from line (e.g. "Steps:  12%|█▎        | 2/16")
                match = re.search(r"(\d+)%", line)
                if match and progress:
                    pct = int(match.group(1))
                    mapped_progress = progress_base + (pct / 100.0) * progress_scale
                    progress(mapped_progress, desc=f"Velora AI generating: {pct}%")
                elif progress:
                    if "LOAD REPORT" in line or "Downloading" in line:
                        progress(progress_base, desc=line[:50])
        else:
            buffer += char

    process.wait()
    full_output = "\n".join(stdout_lines)

    if process.returncode != 0:
        return None, f"CLI Error: {full_output}"

    if os.path.exists(output_path):
        import soundfile as sf
        import numpy as np
        data, sr = sf.read(output_path)
        std = np.std(data)
        if std < 0.001:
            return None, "Output is silent. Try different reference audio."
        return output_path, f"✅ Generated {len(data)/sr:.1f}s audio"

    import glob
    wavs = glob.glob(os.path.join(BASE_DIR, "infer_cli_*.wav"))
    if wavs:
        latest = max(wavs, key=os.path.getmtime)
        return latest, f"✅ Found: {os.path.basename(latest)}"
    return None, "❌ Output file not found!"

# ─── Tab 1: Standard Clone ───
def clone_voice_tab1(text, ref_text, audio_ref, nfe_step, saved_voice_name, progress=gr.Progress()):
    if not text: return None, "Enter text to generate."
    
    # Check if text is Hindi/Urdu
    if is_hindi_urdu_text(text):
        if saved_voice_name:
            # Look for a matching RVC model
            rvc_models = get_rvc_models()
            target_rvc = normalize_character_name(saved_voice_name).lower()
            rvc_model_name = None
            for m in rvc_models:
                m_name = m.replace(".pth", "").lower()
                if m_name == target_rvc or m_name.replace("_", " ") == target_rvc:
                    rvc_model_name = m
                    break
            
            if rvc_model_name:
                progress(0.1, desc="🌐 Detected Hindi/Urdu & RVC model. Running Hybrid system...")
                base_audio, final_text = generate_native_pronunciation_audio(text, saved_voice_name, progress)
                if base_audio:
                    path, log = run_rvc_conversion(base_audio, rvc_model_name, pitch=0)
                    if path:
                        progress(1.0)
                        return path, f"✅ Voice converted using Hybrid RVC model '{rvc_model_name}' for perfect pronunciation.\n{log}"
            
            # Fall back to F5-TTS with warning if no RVC model is found
            warning_msg = f"⚠️ Detected Hindi/Urdu. For perfect native pronunciation, please train/place an RVC model (.pth) matching the name '{saved_voice_name}' in the 'rvc_models' folder.\nFalling back to zero-shot F5-TTS..."
            progress(0.1, desc="Processing reference with F5-TTS fallback...")
            path, log = run_f5tts(text, audio_ref, ref_text, progress=progress, progress_base=0.1, progress_scale=0.85, nfe_step=nfe_step)
            progress(1.0)
            return path, f"{warning_msg}\n{log}"
            
    # Default F5-TTS flow
    if not audio_ref: return None, "Upload a reference audio."
    progress(0.1, desc="Processing reference...")
    path, log = run_f5tts(text, audio_ref, ref_text, progress=progress, progress_base=0.1, progress_scale=0.85, nfe_step=nfe_step)
    progress(1.0)
    return path, log


# ─── Tab 2: Dramatic Story Mode ───
NARRATOR_VOICES = {
    "Guy (Passionate Male)": "en-US-GuyNeural",
    "Christopher (Authority Male)": "en-US-ChristopherNeural",
    "Andrew (Confident Male)": "en-US-AndrewNeural",
    "Eric (Rational Male)": "en-US-EricNeural",
    "Brian (Casual Male)": "en-US-BrianNeural",
    "Jenny (Friendly Female)": "en-US-JennyNeural",
    "Aria (Confident Female)": "en-US-AriaNeural",
    "Ava (Expressive Female)": "en-US-AvaNeural",
    "Ryan (British Male)": "en-GB-RyanNeural",
    "Sonia (British Female)": "en-GB-SoniaNeural",
}

def dramatic_clone(text, saved_voice_name, narrator_style, progress=gr.Progress()):
    if not text:
        return None, None, "Enter a story script."
    if not saved_voice_name:
        return None, None, "Select a saved voice from your library first."

    log_lines = []

    # Step 1: Generate emotional narration via edge-tts
    progress(0.1, desc="Step 1: Generating dramatic narration...")
    
    # Check if text is Hindi/Urdu and map to correct neural voice
    if is_hindi_urdu_text(text):
        lang = detect_script_language(text)
        is_female = "female" in narrator_style.lower()
        if lang == "urdu":
            voice_id = "ur-PK-UzmaNeural" if is_female else "ur-PK-AsadNeural"
        else:
            voice_id = "hi-IN-SwaraNeural" if is_female else "hi-IN-MadhurNeural"
            
        final_text = text
        if lang == "hindi" and not any('\u0900' <= c <= '\u097F' for c in text):
            try:
                from transliterate import roman_to_devanagari
                final_text = roman_to_devanagari(text)
                log_lines.append(f"🔄 Transliterated Roman script for pronunciation mapping.")
            except Exception:
                final_text = text
        log_lines.append(f"🌐 Detected Hindi/Urdu script. Routed base narration to: {voice_id}")
    else:
        voice_id = NARRATOR_VOICES.get(narrator_style, "en-US-GuyNeural")
        final_text = text

    emotion_path = os.path.join(TEMP_DIR, "emotion_base.mp3")
    ok, err = run_edge_tts(final_text, voice_id, emotion_path)
    if not ok:
        return None, None, f"❌ Edge-TTS failed: {err}"
    log_lines.append(f"Step 1: ✅ Emotional narration generated ({narrator_style})")

    # Step 2: Clone into anime voice
    progress(0.3, desc="Step 2: Cloning into anime voice...")
    voice_audio, voice_text = load_voice(saved_voice_name)
    
    # Try using RVC if it is Hindi/Urdu and a matching RVC model is found
    if is_hindi_urdu_text(text):
        rvc_models = get_rvc_models()
        target_rvc = normalize_character_name(saved_voice_name).lower()
        rvc_model_name = None
        for m in rvc_models:
            m_name = m.replace(".pth", "").lower()
            if m_name == target_rvc or m_name.replace("_", " ") == target_rvc:
                rvc_model_name = m
                break
                
        if rvc_model_name:
            progress(0.4, desc="Step 2: Performing RVC voice conversion on emotional base...")
            clone_path, rvc_log = run_rvc_conversion(emotion_path, rvc_model_name, pitch=0)
            if clone_path:
                log_lines.append(f"Step 2: ✅ Successfully converted emotional base to '{saved_voice_name}' using RVC model '{rvc_model_name}'.")
                progress(1.0)
                return emotion_path, clone_path, "\n".join(log_lines)
            else:
                log_lines.append(f"Step 2: ❌ RVC Conversion failed: {rvc_log}")
        else:
            log_lines.append(f"Step 2: ⚠️ No matching RVC model found for '{saved_voice_name}' (place matching .pth in rvc_models/).")
            log_lines.append("Falling back to zero-shot F5-TTS...")

    if not voice_audio:
        log_lines.append(f"Step 2: ⚠️ Voice '{saved_voice_name}' audio not found. Showing emotion base only.")
        return emotion_path, None, "\n".join(log_lines)

    clone_path, clone_log = run_f5tts(text, voice_audio, voice_text, "dramatic_clone.wav", progress=progress, progress_base=0.3, progress_scale=0.65)
    log_lines.append(f"Step 2: {clone_log}")
    progress(1.0)
    return emotion_path, clone_path, "\n".join(log_lines)


# ─── Tab 3: Hindi/Urdu ───
def generate_hindi(text, voice_id, use_transliteration, speed, pitch, progress=gr.Progress()):
    if not text:
        return None, "Enter some text."

    status = []
    final_text = text

    if use_transliteration:
        has_devanagari = any('\u0900' <= c <= '\u097F' for c in text)
        if not has_devanagari:
            from transliterate import roman_to_devanagari
            final_text = roman_to_devanagari(text)
            status.append(f"🔄 Transliterated to: {final_text}")
        else:
            status.append("Text already in Devanagari.")

    output_path = os.path.join(TEMP_DIR, "hindi_output.mp3")

    rate_arg = f"{speed:+d}%" if speed != 0 else None
    pitch_arg = f"{pitch:+d}Hz" if pitch != 0 else None

    progress(0.5, desc="Generating voice...")
    ok, err = run_edge_tts(final_text, voice_id, output_path, rate=rate_arg, pitch=pitch_arg)
    if not ok:
        return None, f"❌ Error: {err}"

    status.append("✅ Generated successfully!")
    progress(1.0)
    return output_path, "\n".join(status)

# ─── Extract Text (Whisper) ───
def extract_text_fn(audio_path, progress=gr.Progress()):
    if not audio_path: return "Upload an audio file first!"
    try:
        trimmed = os.path.join(TEMP_DIR, "extract_temp.wav")
        audio = AudioSegment.from_file(audio_path)
        if len(audio) > 8000: audio = audio[:8000]
        audio.export(trimmed, format="wav")
        progress(0.4, desc="Loading Whisper...")
        import torch
        from transformers import pipeline
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base",
                        device=device, torch_dtype=dtype)
        progress(0.7, desc="Transcribing...")
        result = pipe(trimmed, chunk_length_s=30, generate_kwargs={"task": "transcribe"})
        text = result['text'].strip()
        del pipe
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return text
    except Exception as e:
        return f"Error: {str(e)}"

# ─── Tab 4: Multi-Voice Podcast ───
import re

def parse_podcast_script(script_text):
    """Parse a script like 'NARUTO: Hey! \n LUFFY: Yo!' into [(name, line), ...]"""
    lines = []
    for raw_line in script_text.strip().splitlines():
        raw_line = raw_line.strip()
        if not raw_line or raw_line.startswith("#"):
            continue
        # More robust regex to catch variations like [NAME], NAME -, NAME says:
        match = re.match(r'^\[?([A-Za-z0-9_ ]+?)\]?\s*(?:[:：\-]|says:?)\s*(.+)$', raw_line, re.IGNORECASE)
        if match:
            name = normalize_character_name(match.group(1))
            dialogue = match.group(2).strip()
            if dialogue:
                lines.append((name, dialogue))
        else:
            # Try splitting by first colon as a fallback
            if ":" in raw_line or "：" in raw_line:
                sep = ":" if ":" in raw_line else "："
                parts = raw_line.split(sep, 1)
                name = normalize_character_name(parts[0])
                dialogue = parts[1].strip()
                if name and dialogue:
                    lines.append((name, dialogue))
            else:
                continue
    return lines

def generate_podcast(script_text, pause_ms, progress=gr.Progress()):
    if not script_text.strip():
        return None, "Write a script first."

    parsed = parse_podcast_script(script_text)
    if not parsed:
        return None, "❌ Could not parse script. Use format:\nNARUTO: Hey Luffy!\nLUFFY: Hey Naruto!"

    # Collect unique character names
    characters = list(dict.fromkeys([name for name, _ in parsed]))
    saved = get_saved_voices()

    # Match characters to saved voices
    voice_map = {}
    missing = []
    for char in characters:
        matched = match_saved_voice_name(char, saved)
        if matched:
            voice_map[char] = matched
        else:
            missing.append(char)

    if missing:
        return None, (
            f"❌ These characters have no matching saved voice:\n"
            f"  {', '.join(missing)}\n\n"
            f"Your saved voices: {', '.join(saved)}\n\n"
            f"Character names in your script must match saved voice names.\n"
            f"Go to the Voice Cloner tab to save voices first."
        )

    log_lines = []
    log_lines.append(f"📋 Parsed {len(parsed)} lines from {len(characters)} characters")
    for char in characters:
        log_lines.append(f"  {char} → voice '{voice_map[char]}'")

    # Check if there are any Hindi/Urdu lines in the script
    has_hindi_urdu = any(is_hindi_urdu_text(dialogue) for _, dialogue in parsed)

    # Generate each line
    audio_segments = []
    pause = AudioSegment.silent(duration=int(pause_ms))

    if not has_hindi_urdu:
        log_lines.append("⚡ Running optimized single-call batch generation...")
        progress(0.1, desc="Preparing voices & transcribing references...")
        
        # Build voices dictionary and script with tags
        import tomli_w
        voices_dict = {}
        gen_text_parts = []
        
        for char, dialogue in parsed:
            voice_name = voice_map[char]
            voice_audio, voice_text = load_voice(voice_name)
            if not voice_audio:
                continue
                
            # If no reference text, auto-transcribe it
            if not voice_text or not voice_text.strip():
                class DummyProgress:
                    def __call__(self, *args, **kwargs): pass
                log_lines.append(f"🔍 Transcribing reference audio for '{voice_name}'...")
                voice_text = extract_text_fn(voice_audio, progress=DummyProgress())
                if voice_text.startswith("Error"):
                    log_lines.append(f"⚠️ Failed to transcribe reference for {voice_name}: {voice_text}")
                    voice_text = ""
            
            # Normalize tag name for F5-TTS (alphanumeric only)
            tag_name = re.sub(r'[^A-Za-z0-9]', '', voice_name)
            
            # Add to voices_dict
            voices_dict[tag_name] = {
                "ref_audio": voice_audio,
                "ref_text": voice_text.strip()
            }
            gen_text_parts.append(f"[{tag_name}] {dialogue.strip()}")
            
        config_path = os.path.join(BASE_DIR, "inference_config.toml")
        config_dict = {
            "model": "F5TTS_Base",
            "speed": 1.0,
            "nfe_step": 10,
            "output_dir": BASE_DIR,
            "output_file": "podcast_output.wav",
            "voices": voices_dict,
            "gen_text": " ".join(gen_text_parts)
        }
        
        with open(config_path, "wb") as f:
            tomli_w.dump(config_dict, f)
            
        # Run F5-TTS once
        progress(0.3, desc="Generating entire podcast in a single run...")
        env = os.environ.copy()
        
        # CPU Threading optimizations to prevent thread contention and speed up inference on CPU
        import multiprocessing
        import torch
        cores = str(max(1, multiprocessing.cpu_count() // 2))
        env.update({
            "TEMP": TEMP_DIR,
            "TMP": TEMP_DIR,
            "NUMBA_DISABLE_JIT": "0",
            "HF_HOME": os.environ["HF_HOME"],
            "PYTHONIOENCODING": "utf-8",
            "OMP_NUM_THREADS": cores,
            "MKL_NUM_THREADS": cores,
            "OPENBLAS_NUM_THREADS": cores,
            "NUMEXPR_NUM_THREADS": cores,
            "VECLIB_MAXIMUM_THREADS": cores
        })
        
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
                        
        process = subprocess.Popen(
            [F5_TTS_PYTHON_EXE, "-m", "f5_tts.infer.infer_cli", "-c", config_path, "-o", BASE_DIR, "-w", "podcast_output.wav", "--device", device_str],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            env=env,
            bufsize=1
        )
        
        stdout_lines = []
        buffer = ""
        while True:
            char = process.stdout.read(1)
            if not char:
                break
            if char in ('\r', '\n'):
                line = buffer.strip()
                buffer = ""
                if line:
                    stdout_lines.append(line)
                    match = re.search(r"(\d+)%", line)
                    if match:
                        pct = int(match.group(1))
                        progress(0.3 + (pct / 100.0) * 0.65, desc=f"Generating podcast: {pct}%")
            else:
                buffer += char
                
        process.wait()
        full_output = "\n".join(stdout_lines)
        
        if process.returncode != 0:
            return None, f"CLI Error: {full_output}"
            
        podcast_output_path = os.path.join(BASE_DIR, "podcast_output.wav")
        if os.path.exists(podcast_output_path):
            log_lines.append("✅ Podcast generated successfully!")
            progress(1.0)
            audio = AudioSegment.from_file(podcast_output_path)
            return podcast_output_path, f"✅ Generated successfully! ({len(audio)/1000:.1f}s total)\n" + "\n".join(log_lines)
        else:
            return None, "❌ Output file not found!"

    else:
        # Existing sequential logic for Hindi/Urdu support
        for i, (char, dialogue) in enumerate(parsed):
            progress((i + 1) / len(parsed), desc=f"Generating line {i+1}/{len(parsed)}: {char}...")
            log_lines.append(f"\n🎙️ [{i+1}/{len(parsed)}] {char}: \"{dialogue[:50]}...\"")

            voice_name = voice_map[char]
            voice_audio, voice_text = load_voice(voice_name)
            if not voice_audio:
                log_lines.append(f"  ⚠️ Audio file missing for '{voice_name}', skipping.")
                continue

            out_name = f"podcast_line_{i}.wav"
            
            # Hybrid System: RVC for Hindi/Urdu, F5-TTS otherwise
            if is_hindi_urdu_text(dialogue):
                log_lines.append(f"  🌐 Detected Hindi/Urdu. Routing through Hybrid System (MS Neural -> RVC)...")
                base_audio, text = generate_native_pronunciation_audio(dialogue, voice_name, progress)
                if not base_audio:
                    log_lines.append(f"  ❌ Failed to generate MS Neural base audio.")
                    continue
                    
                # Find RVC model for this character
                rvc_models = get_rvc_models()
                target_rvc = normalize_character_name(voice_name).lower()
                rvc_model_name = None
                for m in rvc_models:
                    m_name = m.replace(".pth", "").lower()
                    if m_name == target_rvc or m_name.replace("_", " ") == target_rvc:
                        rvc_model_name = m
                        break
                
                if not rvc_model_name:
                    log_lines.append(f"  ❌ RVC model not found for '{voice_name}'. Train a model first for native pronunciation.")
                    continue
                    
                path, gen_log = run_rvc_conversion(base_audio, rvc_model_name, pitch=0)
            else:
                # Normal English -> F5-TTS
                base = i / len(parsed)
                scale = 1.0 / len(parsed)
                path, gen_log = run_f5tts(dialogue, voice_audio, voice_text, output_name=out_name, progress=progress, progress_base=base, progress_scale=scale)

            if path and os.path.exists(path):
                # Apply short fade-in/fade-out to prevent clicks/pops and smooth transitions
                seg = AudioSegment.from_file(path).fade_in(50).fade_out(150)
                audio_segments.append(seg)
                log_lines.append(f"  ✅ {len(seg)/1000:.1f}s generated")
            else:
                log_lines.append(f"  ❌ Failed: {gen_log}")

        if not audio_segments:
            return None, "\n".join(log_lines) + "\n\n❌ No audio was generated."

        # Stitch together with brief crossfades and pauses
        log_lines.append(f"\n🔗 Stitching {len(audio_segments)} segments...")
        final = audio_segments[0]
        for seg in audio_segments[1:]:
            # Add smooth natural pause before crossfading
            final += pause
            cf = min(50, len(final), len(seg))
            final = final.append(seg, crossfade=cf)

        output_path = os.path.join(BASE_DIR, "podcast_output.wav")
        final.export(output_path, format="wav")

        log_lines.append(f"✅ Final podcast: {len(final)/1000:.1f}s total")
        return output_path, "\n".join(log_lines)

# ─── Audio Editor Functions ───
def edit_audio_trim(audio_path, start_s, end_s):
    if not audio_path: return None, "Upload audio first."
    try:
        audio = AudioSegment.from_file(audio_path)
        start_ms, end_ms = int(start_s * 1000), int(end_s * 1000)
        trimmed = audio[start_ms:end_ms]
        out = os.path.join(BASE_DIR, "edited_audio.wav")
        trimmed.export(out, format="wav")
        return out, f"✅ Trimmed: Kept {start_s}s to {end_s}s"
    except Exception as e:
        return None, f"❌ Error: {e}"

def edit_audio_cut(audio_path, start_s, end_s):
    if not audio_path: return None, "Upload audio first."
    try:
        audio = AudioSegment.from_file(audio_path)
        start_ms, end_ms = int(start_s * 1000), int(end_s * 1000)
        cut = audio[:start_ms] + audio[end_ms:]
        out = os.path.join(BASE_DIR, "edited_audio.wav")
        cut.export(out, format="wav")
        return out, f"✅ Cut: Removed {start_s}s to {end_s}s"
    except Exception as e:
        return None, f"❌ Error: {e}"

def edit_audio_replace(audio_path, start_s, end_s, text, voice_name, progress=gr.Progress()):
    if not audio_path: return None, "Upload audio first."
    if not text: return None, "Enter text to generate."
    if not voice_name: return None, "Select a voice."
    try:
        audio = AudioSegment.from_file(audio_path)
        start_ms, end_ms = int(start_s * 1000), int(end_s * 1000)
        
        voice_audio, voice_text = load_voice(voice_name)
        if not voice_audio:
            return None, f"❌ Audio file missing for '{voice_name}'"
            
        progress(0.3, desc="Generating new segment...")
        new_path, gen_log = run_f5tts(text, voice_audio, voice_text, output_name="replacement.wav")
        if not new_path or not os.path.exists(new_path):
            return None, f"❌ Generation failed: {gen_log}"
            
        new_seg = AudioSegment.from_file(new_path)
        final = audio[:start_ms] + new_seg + audio[end_ms:]
        
        out = os.path.join(BASE_DIR, "edited_audio.wav")
        final.export(out, format="wav")
        return out, f"✅ Replaced {start_s}s to {end_s}s with new generated audio."
    except Exception as e:
        return None, f"❌ Error: {e}"

# ─── ML FEATURE: Audio Dataset Preprocessing ───
TRAINING_DIR = os.path.join(BASE_DIR, "training_data")
os.makedirs(TRAINING_DIR, exist_ok=True)

def remove_silence_from_audio(audio, min_silence_len=600, silence_thresh=-40, keep_silence=200):
    if silence_thresh is None:
        silence_thresh = -40
    nonsilent_ranges = detect_nonsilent(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    if not nonsilent_ranges:
        return audio
    cleaned = AudioSegment.empty()
    for start, end in nonsilent_ranges:
        start = max(0, start - keep_silence)
        end = min(len(audio), end + keep_silence)
        cleaned += audio[start:end]
    return cleaned


def reduce_background_noise(audio, cutoff_freq=120):
    try:
        import noisereduce as nr
        import numpy as np
        
        samples = np.array(audio.get_array_of_samples())
        
        # Perform noise reduction on mono sample array
        reduced_noise = nr.reduce_noise(y=samples, sr=audio.frame_rate, prop_decrease=0.8)
        
        # Dynamically map numpy type to match original sample width (bit depth)
        if audio.sample_width == 2:
            dtype = np.int16
        elif audio.sample_width == 4:
            dtype = np.int32
        elif audio.sample_width == 1:
            dtype = np.int8
        else:
            dtype = np.int16
            
        new_audio = audio._spawn(reduced_noise.astype(dtype).tobytes())
        return new_audio
    except Exception:
        return audio.high_pass_filter(cutoff_freq)


def preprocess_training_audio(audio_path, chunk_seconds=10, normalize_db=-20.0, progress=gr.Progress()):
    """Real ML data pipeline: chunk, normalize, and clean audio for model training."""
    if not audio_path:
        return None, "Upload an audio file first."
    try:
        progress(0.1, desc="Loading raw audio...")
        audio = AudioSegment.from_file(audio_path)
        original_duration = len(audio) / 1000.0

        # Convert to mono and resample to 16kHz immediately to optimize downstream pipeline speed and prevent crashes
        progress(0.2, desc="Resampling to mono 16kHz...")
        audio = audio.set_channels(1).set_frame_rate(16000)

        progress(0.3, desc="Normalizing volume levels...")
        change_in_dBFS = normalize_db - audio.dBFS
        audio = audio.apply_gain(change_in_dBFS)

        progress(0.45, desc="Filtering background static noise...")
        audio = reduce_background_noise(audio)
        
        progress(0.55, desc="Cutting out silent dead air...")
        audio = remove_silence_from_audio(audio)


        chunk_ms = int(chunk_seconds * 1000)
        progress(0.7, desc="Chunking into training segments...")
        chunks = [audio[i:i + chunk_ms] for i in range(0, len(audio), chunk_ms)]
        chunks = [c for c in chunks if len(c) >= int(chunk_seconds * 1000 * 0.6)]

        if not chunks:
            return None, "❌ Preprocessing Error: no valid audio chunks found after cleanup."

        existing_sessions = [d for d in os.listdir(TRAINING_DIR) if os.path.isdir(os.path.join(TRAINING_DIR, d))]
        session_dir = os.path.join(TRAINING_DIR, f"session_{len(existing_sessions):03d}")
        os.makedirs(session_dir, exist_ok=True)

        progress(0.85, desc="Exporting clean training chunks...")
        for i, chunk in enumerate(chunks):
            chunk.export(os.path.join(session_dir, f"chunk_{i:03d}.wav"), format="wav")

        log = (
            f"✅ Audio Dataset Preprocessed!\n"
            f"📊 Original Duration: {original_duration:.1f}s\n"
            f"🔊 Normalized to: {normalize_db} dBFS\n"
            f"🎵 Resampled to: 16kHz Mono\n"
            f"✂️ Cleaned silence and noise\n"
            f"✂️ Created {len(chunks)} training chunks ({chunk_seconds}s each)\n"
            f"📁 Saved to: {session_dir}"
        )
        progress(1.0)
        return session_dir, log
    except Exception as e:
        return None, f"❌ Preprocessing Error: {e}"

def analyze_voice_similarity(audio_a, audio_b, progress=gr.Progress()):
    """Real ML: Compare two audio files using Whisper encoder embeddings + cosine similarity."""
    if not audio_a or not audio_b:
        return "Upload both audio files to compare."
    try:
        progress(0.2, desc="Loading Whisper encoder...")
        import torch
        import numpy as np
        from transformers import WhisperProcessor, WhisperModel

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        model = WhisperModel.from_pretrained("openai/whisper-base").to(device).to(dtype)

        def get_embedding(path):
            audio = AudioSegment.from_file(path).set_channels(1).set_frame_rate(16000)
            if len(audio) > 15000:
                audio = audio[:15000]
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
            inputs = processor(samples, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(device).to(dtype)
            with torch.no_grad():
                encoder_out = model.encoder(input_features)
                embedding = encoder_out.last_hidden_state.mean(dim=1).squeeze()
            return embedding

        progress(0.5, desc="Extracting voice embeddings...")
        emb_a = get_embedding(audio_a)
        progress(0.7, desc="Comparing voice signatures...")
        emb_b = get_embedding(audio_b)

        # Cosine Similarity
        cos_sim = torch.nn.functional.cosine_similarity(emb_a.unsqueeze(0), emb_b.unsqueeze(0)).item()
        similarity_pct = max(0, min(100, cos_sim * 100))

        # Cleanup GPU
        del model, processor, emb_a, emb_b
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        grade = "🟢 Excellent" if similarity_pct > 85 else "🟡 Good" if similarity_pct > 70 else "🔴 Poor"
        progress(1.0)
        return (
            f"🧠 Voice Similarity Analysis\n"
            f"{'='*40}\n"
            f"Cosine Similarity Score: {similarity_pct:.1f}%\n"
            f"Quality Grade: {grade}\n\n"
            f"{'='*40}\n"
            f"If the score is below 70%, consider:\n"
            f"  • Using a longer/cleaner reference audio\n"
            f"  • Fine-tuning the model with more training data\n"
            f"  • Adjusting the pitch shift parameter"
        )
    except Exception as e:
        return f"❌ Analysis Error: {e}"

# ═══════════════════════════════════════
#  GRADIO UI
# ═══════════════════════════════════════
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ─── Global Reset & Base ─── */
* { box-sizing: border-box; }

body, .gradio-container {
    background: #050812 !important;
    font-family: 'Inter', sans-serif !important;
    min-height: 100vh;
}

/* Animated background mesh */
.gradio-container::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
        radial-gradient(ellipse 80% 80% at 20% 10%, rgba(99,102,241,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 60% at 80% 20%, rgba(139,92,246,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 70% 70% at 50% 90%, rgba(6,182,212,0.05) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
    animation: bgPulse 12s ease-in-out infinite alternate;
}

@keyframes bgPulse {
    0% { opacity: 0.7; }
    100% { opacity: 1; }
}

/* ─── Hide Gradio Footer ─── */
footer { display: none !important; }
.share-button { display: none !important; }

/* ─── Scrollbar ─── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.03); border-radius: 3px; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.4); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.7); }

/* ─── Header ─── */
.vs-header {
    text-align: center;
    padding: 40px 20px 32px;
    position: relative;
    overflow: hidden;
}

.vs-header::before {
    content: '';
    position: absolute;
    top: -40px; left: 50%; transform: translateX(-50%);
    width: 600px; height: 200px;
    background: radial-gradient(ellipse, rgba(99,102,241,0.15) 0%, transparent 70%);
    pointer-events: none;
}

.vs-logo-wrap {
    display: inline-flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 12px;
}

.vs-logo-icon {
    width: 52px; height: 52px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6, #06b6d4);
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 26px;
    box-shadow: 0 0 30px rgba(99,102,241,0.5), 0 0 60px rgba(139,92,246,0.2);
    animation: iconGlow 3s ease-in-out infinite alternate;
    flex-shrink: 0;
}

@keyframes iconGlow {
    0% { box-shadow: 0 0 20px rgba(99,102,241,0.4), 0 0 40px rgba(139,92,246,0.15); }
    100% { box-shadow: 0 0 40px rgba(99,102,241,0.7), 0 0 80px rgba(139,92,246,0.35); }
}

.vs-logo-text {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.4em;
    font-weight: 700;
    background: linear-gradient(135deg, #a5b4fc, #c4b5fd, #67e8f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 1px;
    line-height: 1;
}

.vs-badge {
    display: inline-block;
    background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(139,92,246,0.15));
    border: 1px solid rgba(99,102,241,0.3);
    color: #a5b4fc;
    font-size: 0.7em;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 6px;
}

.vs-subtitle {
    color: rgba(165,180,252,0.6);
    font-size: 0.95em;
    font-weight: 400;
    margin-top: 6px;
    letter-spacing: 0.3px;
}

.vs-divider {
    width: 100%;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,0.3), rgba(139,92,246,0.3), transparent);
    margin-top: 28px;
}

/* ─── Tabs ─── */
.tabs { border: none !important; background: transparent !important; }

.tab-nav {
    background: rgba(15,18,30,0.8) !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
    border-radius: 14px !important;
    padding: 4px !important;
    gap: 2px !important;
    backdrop-filter: blur(20px) !important;
    margin-bottom: 20px !important;
}

.tab-nav button {
    background: transparent !important;
    border: none !important;
    color: rgba(165,180,252,0.55) !important;
    border-radius: 10px !important;
    padding: 9px 16px !important;
    font-size: 0.82em !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.25s ease !important;
    white-space: nowrap;
}

.tab-nav button:hover {
    background: rgba(99,102,241,0.1) !important;
    color: rgba(165,180,252,0.85) !important;
}

.tab-nav button.selected {
    background: linear-gradient(135deg, rgba(99,102,241,0.25), rgba(139,92,246,0.2)) !important;
    color: #a5b4fc !important;
    box-shadow: 0 0 0 1px rgba(99,102,241,0.35), 0 4px 15px rgba(99,102,241,0.15) !important;
}

/* ─── Cards / Panels ─── */
.gr-group, .gr-box, div[data-testid="column"] > div {
    border-radius: 16px !important;
}

/* Glass panel base */
.gr-group {
    background: rgba(15,18,35,0.7) !important;
    border: 1px solid rgba(99,102,241,0.12) !important;
    backdrop-filter: blur(24px) !important;
    -webkit-backdrop-filter: blur(24px) !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.04) !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
}

.gr-group:hover {
    border-color: rgba(99,102,241,0.22) !important;
    box-shadow: 0 8px 40px rgba(0,0,0,0.4), 0 0 0 1px rgba(99,102,241,0.08), inset 0 1px 0 rgba(255,255,255,0.05) !important;
}

/* ─── Labels ─── */
label span, .gr-form label, fieldset legend span {
    color: rgba(196,181,253,0.8) !important;
    font-size: 0.8em !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    text-transform: uppercase !important;
    font-family: 'Inter', sans-serif !important;
}

/* ─── Inputs & Textareas ─── */
input[type="text"], input[type="number"], textarea, .gr-input {
    background: rgba(8,10,22,0.7) !important;
    border: 1px solid rgba(99,102,241,0.18) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9em !important;
    transition: border-color 0.25s, box-shadow 0.25s !important;
    padding: 10px 14px !important;
}

input[type="text"]:focus, input[type="number"]:focus, textarea:focus {
    border-color: rgba(99,102,241,0.5) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.12), 0 0 20px rgba(99,102,241,0.08) !important;
    outline: none !important;
    background: rgba(10,13,28,0.85) !important;
}

textarea { resize: vertical !important; min-height: 80px !important; line-height: 1.6 !important; }

/* ─── Dropdowns ─── */
.gr-dropdown, select, .multiselect {
    background: rgba(8,10,22,0.7) !important;
    border: 1px solid rgba(99,102,241,0.18) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

/* ─── Buttons ─── */
button.gr-button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    transition: all 0.25s ease !important;
    position: relative !important;
    overflow: hidden !important;
    letter-spacing: 0.3px !important;
}

button.gr-button::before {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.06), transparent);
    transition: left 0.5s ease;
}

button.gr-button:hover::before { left: 100%; }

/* Primary button */
button.gr-button.primary, button[variant="primary"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    color: #fff !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.35), 0 0 0 1px rgba(99,102,241,0.5) !important;
    font-size: 0.92em !important;
    padding: 11px 20px !important;
}

button.gr-button.primary:hover, button[variant="primary"]:hover {
    background: linear-gradient(135deg, #818cf8, #a78bfa) !important;
    box-shadow: 0 6px 25px rgba(99,102,241,0.5), 0 0 0 1px rgba(99,102,241,0.7) !important;
    transform: translateY(-1px) !important;
}

button.gr-button.primary:active, button[variant="primary"]:active {
    transform: translateY(0px) !important;
    box-shadow: 0 2px 10px rgba(99,102,241,0.35) !important;
}

/* Secondary button */
button.gr-button.secondary, button[variant="secondary"] {
    background: rgba(99,102,241,0.08) !important;
    border: 1px solid rgba(99,102,241,0.25) !important;
    color: #a5b4fc !important;
    font-size: 0.85em !important;
}

button.gr-button.secondary:hover, button[variant="secondary"]:hover {
    background: rgba(99,102,241,0.15) !important;
    border-color: rgba(99,102,241,0.45) !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.15) !important;
}

/* Stop/Delete button */
button.gr-button.stop, button[variant="stop"] {
    background: rgba(239,68,68,0.08) !important;
    border: 1px solid rgba(239,68,68,0.25) !important;
    color: #f87171 !important;
    font-size: 0.85em !important;
}

button.gr-button.stop:hover, button[variant="stop"]:hover {
    background: rgba(239,68,68,0.15) !important;
    border-color: rgba(239,68,68,0.45) !important;
    box-shadow: 0 4px 15px rgba(239,68,68,0.15) !important;
}

/* Large button sizing */
button.lg, button[size="lg"] {
    padding: 13px 28px !important;
    font-size: 0.95em !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
}

/* ─── Sliders ─── */
input[type="range"] {
    accent-color: #6366f1 !important;
}

.gr-slider .thumb {
    background: #6366f1 !important;
    box-shadow: 0 0 10px rgba(99,102,241,0.6) !important;
}

/* ─── Audio Player ─── */
.gr-audio, audio {
    background: rgba(8,10,22,0.7) !important;
    border: 1px solid rgba(99,102,241,0.18) !important;
    border-radius: 12px !important;
    padding: 8px !important;
}

.gr-audio:hover {
    border-color: rgba(99,102,241,0.35) !important;
    box-shadow: 0 0 20px rgba(99,102,241,0.08) !important;
}

/* ─── Markdown & Typography ─── */
.gr-markdown, .markdown-text {
    color: rgba(203,213,225,0.85) !important;
    font-family: 'Inter', sans-serif !important;
    line-height: 1.65 !important;
}

.gr-markdown h1, .gr-markdown h2, .gr-markdown h3 {
    color: #c4b5fd !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.3px !important;
    margin-top: 12px !important;
    margin-bottom: 8px !important;
}

.gr-markdown h3 { color: #a5b4fc !important; font-size: 1em !important; }

.gr-markdown strong { color: #c4b5fd !important; font-weight: 600 !important; }

.gr-markdown code {
    background: rgba(99,102,241,0.12) !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    color: #a5b4fc !important;
    padding: 2px 6px !important;
    border-radius: 5px !important;
    font-size: 0.88em !important;
}

.gr-markdown hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,0.25), transparent) !important;
    margin: 14px 0 !important;
}

/* ─── Rows ─── */
.gr-row { gap: 16px !important; }
.gr-column { gap: 12px !important; }

/* ─── Progress bar ─── */
.progress-bar, .progress-bar-fill {
    background: linear-gradient(90deg, #6366f1, #8b5cf6, #06b6d4) !important;
    border-radius: 4px !important;
}

/* ─── Status indicators ─── */
.gr-textbox {
    background: rgba(8,10,22,0.7) !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
    border-radius: 10px !important;
}

/* ─── Neon accent section headers ─── */
.section-title {
    background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(139,92,246,0.05));
    border-left: 3px solid rgba(99,102,241,0.7);
    padding: 8px 14px;
    border-radius: 0 8px 8px 0;
    color: #a5b4fc;
    font-weight: 600;
    font-size: 0.9em;
    margin-bottom: 12px;
}

/* ─── Checkbox & Radio ─── */
input[type="checkbox"] { accent-color: #6366f1 !important; }
input[type="radio"] { accent-color: #6366f1 !important; }

/* ─── File upload ─── */
.gr-file-upload {
    background: rgba(8,10,22,0.5) !important;
    border: 2px dashed rgba(99,102,241,0.2) !important;
    border-radius: 12px !important;
    color: rgba(165,180,252,0.6) !important;
    transition: border-color 0.3s, background 0.3s !important;
}

.gr-file-upload:hover {
    border-color: rgba(99,102,241,0.45) !important;
    background: rgba(99,102,241,0.05) !important;
}

/* ─── Tooltip / Info ─── */
.info-text, .gr-info {
    color: rgba(165,180,252,0.5) !important;
    font-size: 0.78em !important;
}

/* ─── Tab content area ─── */
.tabitem {
    background: transparent !important;
    border: none !important;
    padding-top: 8px !important;
}

/* ─── Glow pulse for active elements ─── */
@keyframes glowPulse {
    0%, 100% { box-shadow: 0 0 8px rgba(99,102,241,0.2); }
    50% { box-shadow: 0 0 20px rgba(99,102,241,0.45); }
}

/* ─── Stat badge style for key info in markdown ─── */
.gr-markdown p > strong:first-child {
    display: inline-block;
    color: #a5b4fc !important;
}
"""

header_html = """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
@keyframes shimmer {
    0% { background-position: -400px 0; }
    100% { background-position: 400px 0; }
}
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-4px); }
}
@keyframes orb1 {
    0%, 100% { transform: translate(0,0) scale(1); opacity: 0.4; }
    33% { transform: translate(30px, -20px) scale(1.1); opacity: 0.6; }
    66% { transform: translate(-20px, 15px) scale(0.95); opacity: 0.35; }
}
@keyframes orb2 {
    0%, 100% { transform: translate(0,0) scale(1); opacity: 0.3; }
    33% { transform: translate(-25px, 20px) scale(1.05); opacity: 0.5; }
    66% { transform: translate(20px, -15px) scale(0.9); opacity: 0.25; }
}
</style>

<div style="
    text-align: center;
    padding: 44px 24px 36px;
    position: relative;
    overflow: hidden;
    border-bottom: 1px solid rgba(99,102,241,0.12);
    margin-bottom: 8px;
">
  <!-- Background orbs -->
  <div style="position:absolute;top:-60px;left:15%;width:320px;height:320px;border-radius:50%;
       background:radial-gradient(circle,rgba(99,102,241,0.12) 0%,transparent 70%);
       animation: orb1 10s ease-in-out infinite; pointer-events:none;"></div>
  <div style="position:absolute;top:-40px;right:10%;width:260px;height:260px;border-radius:50%;
       background:radial-gradient(circle,rgba(139,92,246,0.10) 0%,transparent 70%);
       animation: orb2 13s ease-in-out infinite; pointer-events:none;"></div>
  <div style="position:absolute;bottom:-30px;left:50%;transform:translateX(-50%);width:500px;height:200px;border-radius:50%;
       background:radial-gradient(ellipse,rgba(6,182,212,0.06) 0%,transparent 70%);
       pointer-events:none;"></div>

  <!-- Badge -->
  <div style="display:inline-block;
       background:linear-gradient(135deg,rgba(99,102,241,0.12),rgba(139,92,246,0.12));
       border:1px solid rgba(99,102,241,0.3);
       color:#a5b4fc;font-size:0.68em;font-weight:700;
       padding:4px 14px;border-radius:20px;letter-spacing:2px;
       text-transform:uppercase;margin-bottom:14px;
       box-shadow:0 0 15px rgba(99,102,241,0.15);
       font-family:'Inter',sans-serif;">
    ✦ AI Voice Studio ✦
  </div>

  <!-- Logo -->
  <div style="animation: float 5s ease-in-out infinite;">
    <div style="display:inline-flex;align-items:center;gap:16px;margin-bottom:14px;">
      <div style="
           width:56px;height:56px;
           background:linear-gradient(135deg,#6366f1,#8b5cf6,#06b6d4);
           border-radius:16px;
           display:flex;align-items:center;justify-content:center;
           font-size:28px;flex-shrink:0;
           box-shadow:0 0 30px rgba(99,102,241,0.55),0 0 60px rgba(139,92,246,0.25),inset 0 1px 0 rgba(255,255,255,0.2);">
        🎙️
      </div>
      <div style="text-align:left;">
        <div style="
             font-family:'Space Grotesk',sans-serif;
             font-size:2.2em;font-weight:700;
             background:linear-gradient(135deg,#a5b4fc 0%,#c4b5fd 40%,#67e8f9 100%);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;
             background-clip:text;letter-spacing:0.5px;line-height:1.05;">
          VELORA VOICE STUDIO
        </div>
        <div style="
             font-size:0.7em;font-weight:400;letter-spacing:0.2px;
             background:linear-gradient(90deg,rgba(165,180,252,0.5),rgba(196,181,253,0.5),rgba(103,232,249,0.5));
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;
             background-clip:text;font-family:'Inter',sans-serif;margin-top:3px;">
          Clone · Create · Elevate
        </div>
      </div>
    </div>
  </div>

  <!-- Feature pills -->
  <div style="display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin-top:8px;">
    <span style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.22);
         color:rgba(165,180,252,0.8);padding:5px 12px;border-radius:20px;
         font-size:0.75em;font-weight:500;font-family:'Inter',sans-serif;">🎭 Zero-Shot Voice Cloning</span>
    <span style="background:rgba(139,92,246,0.1);border:1px solid rgba(139,92,246,0.22);
         color:rgba(196,181,253,0.8);padding:5px 12px;border-radius:20px;
         font-size:0.75em;font-weight:500;font-family:'Inter',sans-serif;">🌏 Hindi · Urdu Pronunciation</span>
    <span style="background:rgba(6,182,212,0.1);border:1px solid rgba(6,182,212,0.22);
         color:rgba(103,232,249,0.8);padding:5px 12px;border-radius:20px;
         font-size:0.75em;font-weight:500;font-family:'Inter',sans-serif;">🎙️ Multi-Voice Podcasts</span>
    <span style="background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.2);
         color:rgba(110,231,183,0.8);padding:5px 12px;border-radius:20px;
         font-size:0.75em;font-weight:500;font-family:'Inter',sans-serif;">🧠 AI Model Training</span>
  </div>

  <!-- Divider line -->
  <div style="width:100%;height:1px;
       background:linear-gradient(90deg,transparent,rgba(99,102,241,0.35),rgba(139,92,246,0.35),rgba(6,182,212,0.2),transparent);
       margin-top:30px;"></div>
</div>
"""

# Build the dark glassmorphism theme (Gradio 6.0: theme must be passed to launch())
_dark_theme = gr.themes.Base(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.purple,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Inter"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
).set(
    body_background_fill="#050812",
    body_text_color="#e2e8f0",
    block_background_fill="rgba(15,18,35,0.7)",
    block_border_color="rgba(99,102,241,0.12)",
    block_label_background_fill="transparent",
    block_label_text_color="rgba(165,180,252,0.75)",
    block_title_text_color="#c4b5fd",
    input_background_fill="rgba(8,10,22,0.7)",
    input_border_color="rgba(99,102,241,0.18)",
    input_placeholder_color="rgba(148,163,184,0.35)",
    button_primary_background_fill="linear-gradient(135deg, #6366f1, #8b5cf6)",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="rgba(99,102,241,0.08)",
    button_secondary_text_color="#a5b4fc",
    border_color_primary="rgba(99,102,241,0.2)",
    shadow_spread="8px",
    block_shadow="0 8px 32px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.04)",
    checkbox_background_color="rgba(8,10,22,0.7)",
    slider_color="#6366f1",
    table_border_color="rgba(99,102,241,0.12)",
    table_odd_background_fill="rgba(15,18,35,0.4)",
    table_even_background_fill="rgba(8,10,22,0.3)",
)

with gr.Blocks(title="🎙️ Velora Voice Studio") as interface:
    gr.HTML(header_html)

    with gr.Tabs():
        # ─── TAB 1: Voice Cloner ───
        with gr.TabItem("🎭 Voice Cloner"):
            gr.Markdown("Upload any voice clip → the AI clones it and speaks your text in that voice.")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📂 Voice Library")
                    saved_dd = gr.Dropdown(choices=get_saved_voices(), label="Saved Voices", interactive=True, allow_custom_value=True)
                    with gr.Row():
                        load_btn = gr.Button("📂 Load", size="sm")
                        refresh_btn = gr.Button("🔄 Refresh", size="sm")
                        del_btn = gr.Button("🗑️ Delete", size="sm", variant="stop")
                    gr.Markdown("---")
                    gr.Markdown("### 💾 Save Voice")
                    voice_name = gr.Textbox(label="Name", placeholder="e.g. Gojo_Dramatic")
                    save_btn = gr.Button("💾 Save to Library", variant="primary")
                    lib_status = gr.Textbox(label="Status", interactive=False)

                with gr.Column(scale=2):
                    gen_text1 = gr.Textbox(label="Script to Speak", lines=6, placeholder="Type your story here...")
                    ref_audio1 = gr.Audio(type="filepath", label="Reference Voice (auto-trims to 8s)")
                    with gr.Row():
                        ref_text1 = gr.Textbox(label="Reference Text", lines=2, scale=4,
                            placeholder="Type exact words from the reference audio...")
                        extract_btn1 = gr.Button("🔍 Auto-Extract", variant="secondary", scale=1)
                    nfe_step_slider = gr.Slider(minimum=4, maximum=32, value=10, step=1, label="⚡ Speed vs Quality (Steps)", info="Lower = Faster generation on CPU, Higher = Better quality (Default: 10)")
                    clone_btn1 = gr.Button("🎙️ Generate Clone", variant="primary", size="lg")

            with gr.Row():
                out_audio1 = gr.Audio(label="Generated Audio")
                out_log1 = gr.Textbox(label="Log")

            load_btn.click(fn=load_voice, inputs=[saved_dd], outputs=[ref_audio1, ref_text1])
            extract_btn1.click(fn=extract_text_fn, inputs=[ref_audio1], outputs=[ref_text1])
            clone_btn1.click(fn=clone_voice_tab1, inputs=[gen_text1, ref_text1, ref_audio1, nfe_step_slider, saved_dd], outputs=[out_audio1, out_log1])

        # ─── TAB 2: Dramatic Story Mode ───
        with gr.TabItem("🎬 Dramatic Story Mode", visible=True):
            gr.Markdown("""### How it works:
1. **Step 1:** Microsoft Neural AI creates a dramatic, emotional narration (perfect pronunciation & emotions).
2. **Step 2:** F5-TTS re-generates the same script using your saved anime voice (Gojo, Naruto, etc).
3. You get **two outputs** — pick whichever sounds better!

**Pro tip:** The emotion base alone sounds incredible for YouTube. The anime clone adds character flavor.""")

            with gr.Row():
                with gr.Column():
                    saved_dd2 = gr.Dropdown(choices=get_saved_voices(), label="Select Saved Anime Voice", interactive=True, allow_custom_value=True)
                    narrator_style = gr.Dropdown(
                        choices=list(NARRATOR_VOICES.keys()),
                        label="Emotion Narrator Style", value="Guy (Passionate Male)"
                    )
                    story_text = gr.Textbox(label="Your Story Script", lines=10,
                        placeholder="My daughter went missing five years ago...")
                    dramatic_btn = gr.Button("🎬 Generate Dramatic Voiceover", variant="primary", size="lg")

                with gr.Column():
                    gr.Markdown("### Step 1: Emotional Narration (Microsoft Neural)")
                    emotion_audio = gr.Audio(label="Emotion Base")
                    gr.Markdown("### Step 2: Anime Voice Clone (F5-TTS)")
                    clone_audio = gr.Audio(label="Anime Voice Version")
                    dramatic_log = gr.Textbox(label="Generation Log")

            dramatic_btn.click(fn=dramatic_clone,
                inputs=[story_text, saved_dd2, narrator_style],
                outputs=[emotion_audio, clone_audio, dramatic_log])

        # ─── TAB 3: Multi-Voice Podcast ───
        with gr.TabItem("🎙️ Multi-Voice Podcast"):
            gr.Markdown("""### Create Podcasts with Multiple Anime Voices
Write a script with character names that **match your saved voices**. Each line is generated with the correct voice and stitched into one seamless audio.

**Script Format:**
```
NARUTO: Hey Luffy, what's up man!
LUFFY: Yo Naruto! Just finished eating, I'm pumped!
NARUTO: Wanna go train together?
LUFFY: Let's gooo!
```
⚠️ Character names must **exactly match** your saved voice names (case-insensitive).""")

            with gr.Row():
                with gr.Column():
                    podcast_voices_dd = gr.Dropdown(choices=get_saved_voices(), multiselect=True, label="Your Saved Voices", info="Select the characters you want to use in your podcast script", interactive=True, allow_custom_value=True)
                    podcast_script = gr.Textbox(label="Podcast Script", lines=14,
                        placeholder="NARUTO: Hey Luffy, what's going on?\nLUFFY: Hey Naruto! Just had the best meat ever!\nNARUTO: That sounds awesome, want to spar?\nLUFFY: You're on!")
                    pause_slider = gr.Slider(100, 2000, value=500, step=50,
                        label="Pause Between Lines (ms)", info="How long to pause between each character's line")
                    podcast_btn = gr.Button("🎙️ Generate Full Podcast", variant="primary", size="lg")

                with gr.Column():
                    podcast_audio = gr.Audio(label="Final Podcast Audio")
                    podcast_log = gr.Textbox(label="Generation Log", lines=15)

            podcast_btn.click(fn=generate_podcast,
                inputs=[podcast_script, pause_slider],
                outputs=[podcast_audio, podcast_log])

        # ─── TAB 4: Hindi / Urdu ───
        with gr.TabItem("🌏 Hindi / Urdu"):
            gr.Markdown("""### Perfect Hindi & Urdu Pronunciation
**Fix:** Auto-converts Roman Hindi/Urdu → Devanagari script before generating, so pronunciation is accurate.
- Type **Roman** (kya haal hai) → auto-converts to **Devanagari** (क्या हाल है)
- Or type directly in **Devanagari** for best quality""")

            with gr.Row():
                with gr.Column():
                    hindi_text = gr.Textbox(label="Hindi / Urdu Text", lines=6,
                        placeholder="Hello bhai, kya haal hai? Aaj hum ek bahut hi dilchasp kahani sunenge...")
                    transliterate_toggle = gr.Checkbox(label="🔄 Auto-convert Roman → Devanagari (Recommended!)", value=True)
                    hindi_voice = gr.Dropdown(
                        choices=["hi-IN-MadhurNeural", "hi-IN-SwaraNeural",
                                 "ur-PK-AsadNeural", "ur-PK-UzmaNeural",
                                 "ur-IN-SalmanNeural", "ur-IN-GulNeural"],
                        label="Voice", value="hi-IN-MadhurNeural",
                        info="Madhur=Hindi Male, Swara=Hindi Female, Asad=Urdu Male, Uzma=Urdu Female"
                    )
                    with gr.Row():
                        hindi_speed = gr.Slider(-30, 30, value=0, step=5, label="Speed (%)")
                        hindi_pitch = gr.Slider(-20, 20, value=0, step=2, label="Pitch (Hz)")
                    hindi_btn = gr.Button("🎙️ Generate Hindi/Urdu Voice", variant="primary", size="lg")

                with gr.Column():
                    hindi_audio = gr.Audio(label="Generated Audio")
                    hindi_log = gr.Textbox(label="Status")

            hindi_btn.click(fn=generate_hindi,
                inputs=[hindi_text, hindi_voice, transliterate_toggle, hindi_speed, hindi_pitch],
                outputs=[hindi_audio, hindi_log])

        # ─── TAB 5: Audio Editor ───
        with gr.TabItem("✂️ Audio Editor"):
            gr.Markdown("Upload an audio file (or download a generated one and upload here) to trim, cut, or completely replace a bad segment with a newly generated voice!")
            
            with gr.Row():
                with gr.Column(scale=1):
                    edit_audio_in = gr.Audio(type="filepath", label="Source Audio", interactive=True)
                    start_s = gr.Number(label="Start Time (seconds)", value=0.0)
                    end_s = gr.Number(label="End Time (seconds)", value=5.0)
                    
                    with gr.Row():
                        trim_btn = gr.Button("✂️ Trim (Keep Only Selection)", variant="secondary")
                        cut_btn = gr.Button("🗑️ Cut (Remove Selection)", variant="secondary")
                        
                    gr.Markdown("### Replace Segment")
                    replace_text = gr.Textbox(label="New Text for Segment", lines=2)
                    replace_voice = gr.Dropdown(choices=get_saved_voices(), label="Select Voice for New Segment", interactive=True, allow_custom_value=True)
                    replace_btn = gr.Button("🔄 Replace Segment", variant="primary")
                
                with gr.Column(scale=1):
                    edit_audio_out = gr.Audio(label="Edited Audio")
                    edit_log = gr.Textbox(label="Status Log")
                    
            trim_btn.click(fn=edit_audio_trim, inputs=[edit_audio_in, start_s, end_s], outputs=[edit_audio_out, edit_log])
            cut_btn.click(fn=edit_audio_cut, inputs=[edit_audio_in, start_s, end_s], outputs=[edit_audio_out, edit_log])
            replace_btn.click(fn=edit_audio_replace, inputs=[edit_audio_in, start_s, end_s, replace_text, replace_voice], outputs=[edit_audio_out, edit_log])

        # ─── TAB 6: Voice-to-Voice (RVC) ───
        with gr.TabItem("🎤 Voice-to-Voice (RVC)", visible=True):
            gr.Markdown(f"""### True Emotional Voice Cloning (Speech-to-Speech)
Upload an audio of **you acting out a line**, select a downloaded `.pth` anime character model, and the AI will convert your voice while preserving exactly the timing, emotion, and breath.
*(Models must be placed in `{RVC_MODELS_DIR}`)*""")
            with gr.Row():
                with gr.Column():
                    rvc_in = gr.Audio(type="filepath", label="Input Audio (Your acting/reference)")
                    rvc_model = gr.Dropdown(choices=get_rvc_models(), label="RVC Model (.pth)", interactive=True, allow_custom_value=True)
                    rvc_refresh = gr.Button("🔄 Refresh Models List", size="sm")
                    rvc_pitch = gr.Slider(-24, 24, value=0, step=1, label="Pitch Shift (Semitones)", info="Use +12 for Male->Female, -12 for Female->Male. Leave 0 if same gender.")
                    rvc_btn = gr.Button("🎤 Convert Voice", variant="primary", size="lg")
                with gr.Column():
                    rvc_out = gr.Audio(label="Converted Audio")
                    rvc_log = gr.Textbox(label="Status Log", lines=10)
                    
            rvc_btn.click(fn=run_rvc_conversion, inputs=[rvc_in, rvc_model, rvc_pitch], outputs=[rvc_out, rvc_log])
            rvc_refresh.click(fn=lambda: gr.update(choices=get_rvc_models()), outputs=[rvc_model])

        # ─── TAB 7: Perfect Pronunciation Clone ───
        with gr.TabItem("🌟 Perfect Pronunciation Clone", visible=True):
            gr.Markdown("""### Get Anime Voices with PERFECT Pronunciation
F5-TTS sometimes struggles with pronunciation. This tab fixes that! 
It uses **Edge-TTS (Eric, Guy, etc.)** to generate perfect, native pronunciation, and then uses **RVC** to seamlessly morph that audio into your Anime character's voice.
*(Requires an RVC `.pth` model in `rvc_models/`)*""")
            with gr.Row():
                with gr.Column():
                    perf_text = gr.Textbox(label="Script", lines=6, placeholder="Type perfectly pronounced English here...")
                    perf_neural = gr.Dropdown(choices=list(NARRATOR_VOICES.keys()), label="Base Neural Voice (for acting/pronunciation)", value="Eric (Rational Male)")
                    perf_rvc = gr.Dropdown(choices=get_rvc_models(), label="Target Anime Voice (RVC Model)", interactive=True, allow_custom_value=True)
                    perf_pitch = gr.Slider(-24, 24, value=0, step=1, label="Pitch Shift", info="Match Neural gender to Anime gender. e.g. Male to Female: +12")
                    perf_btn = gr.Button("🌟 Generate Perfect Clone", variant="primary", size="lg")
                with gr.Column():
                    perf_audio = gr.Audio(label="Final Perfect Audio")
                    perf_log = gr.Textbox(label="Status Log")

            def run_perfect_clone(text, neural_voice, rvc_model, pitch, progress=gr.Progress()):
                if not text: return None, "Please enter text."
                if not rvc_model: return None, "Please select an RVC model."
                
                progress(0.2, desc="Generating perfect pronunciation...")
                voice_id = NARRATOR_VOICES.get(neural_voice, "en-US-EricNeural")
                temp_audio = os.path.join(TEMP_DIR, "perf_base.mp3")
                ok, err = run_edge_tts(text, voice_id, temp_audio)
                if not ok:
                    return None, f"❌ Edge-TTS failed: {err}"
                
                progress(0.6, desc="Morphing into Anime Voice (RVC)...")
                final_path, log = run_rvc_conversion(temp_audio, rvc_model, pitch)
                progress(1.0)
                return final_path, log
            
            perf_btn.click(fn=run_perfect_clone, inputs=[perf_text, perf_neural, perf_rvc, perf_pitch], outputs=[perf_audio, perf_log])

        # ─── TAB 8: Voice Training Studio (Real ML) ───
        with gr.TabItem("🧠 Voice Training Studio"):
            gr.Markdown("""### 🧠 AI Model Training Pipeline
This is the **core Machine Learning** feature of the application. Instead of relying on zero-shot cloning (which can sound robotic), you can **train a custom voice model** by feeding it high-quality audio data.

**How it works (Real ML Pipeline):**
1. **Upload** a long audio recording of your target voice (5-10 minutes recommended).
2. **Preprocess** — Our pipeline will automatically normalize volume levels, resample to 16kHz mono (the standard for speech ML models), remove silence, and chunk the audio into clean 10-second training segments.
3. **Analyze** — Use the Voice Quality Analyzer to compare your cloned output vs the original and get a real ML similarity score using Whisper neural embeddings.

*This is the exact same data preprocessing pipeline used in production ML systems at companies like ElevenLabs and OpenAI.*""")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Step 1: Upload Raw Training Audio")
                    train_audio = gr.Audio(type="filepath", label="Raw Training Audio (5-10 min recommended)")
                    chunk_size = gr.Slider(5, 30, value=10, step=1, label="Chunk Size (seconds)", info="Each chunk becomes one training sample")
                    norm_db = gr.Slider(-30, -10, value=-20, step=1, label="Target Volume (dBFS)", info="Normalizes all chunks to this volume level for consistent training")
                    preprocess_btn = gr.Button("⚙️ Preprocess Dataset", variant="primary", size="lg")

                with gr.Column():
                    gr.Markdown("### Preprocessing Results")
                    train_output_dir = gr.Textbox(label="Output Directory", interactive=False)
                    train_log = gr.Textbox(label="Pipeline Log", lines=10)

            preprocess_btn.click(fn=preprocess_training_audio,
                inputs=[train_audio, chunk_size, norm_db],
                outputs=[train_output_dir, train_log])

            gr.Markdown("---")
            gr.Markdown("""### Step 2: Voice Quality Analyzer (Cosine Similarity)
Upload the **original voice** and your **cloned output** to measure how accurate the clone is using real ML metrics.
The system uses **OpenAI Whisper's neural encoder** to extract voice embeddings and computes **cosine similarity** — the same technique used in speaker verification systems.""")

            with gr.Row():
                with gr.Column():
                    sim_audio_a = gr.Audio(type="filepath", label="Audio A: Original Voice")
                    sim_audio_b = gr.Audio(type="filepath", label="Audio B: Cloned Voice")
                    sim_btn = gr.Button("🧠 Analyze Similarity", variant="primary", size="lg")
                with gr.Column():
                    sim_result = gr.Textbox(label="ML Analysis Results", lines=12)

            sim_btn.click(fn=analyze_voice_similarity,
                inputs=[sim_audio_a, sim_audio_b],
                outputs=[sim_result])

    # Global event bindings
    save_btn.click(fn=save_voice, inputs=[voice_name, ref_audio1, ref_text1], outputs=[lib_status, saved_dd, saved_dd2, podcast_voices_dd, replace_voice])
    del_btn.click(fn=delete_voice, inputs=[saved_dd], outputs=[lib_status, saved_dd, saved_dd2, podcast_voices_dd, replace_voice])
    refresh_btn.click(fn=refresh_library, outputs=[saved_dd, saved_dd2, podcast_voices_dd, replace_voice])

if __name__ == "__main__":
    SERVER_NAME = os.environ.get("SERVER_NAME", "127.0.0.1")
    INBROWSER = os.environ.get("INBROWSER", "false").lower() in ("1", "true", "yes")
    print("Launching Advanced Voice Studio...")
    print(f"Saved Voices: {get_saved_voices()}")
    interface.launch(server_name=SERVER_NAME, inbrowser=INBROWSER, css=custom_css, theme=_dark_theme)
