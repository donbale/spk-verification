# Speaker Verification + Transcription (Faster-Whisper + Pyannote)

This is a simple Python project that combines **speaker verification** (using [pyannote.audio](https://github.com/pyannote/pyannote-audio)) with **speech transcription** (using [faster-whisper](https://github.com/SYSTRAN/faster-whisper)).

It allows you to:
- Continuously listen to the microphone  
- Verify whether the speaker matches a reference voice (`yourname.wav`)  
- Transcribe their speech using Whisper  

---

## ⚡ Features
- **Speaker verification** with cosine similarity of embeddings  
- **Fast transcription** via `faster-whisper` (GPU or CPU)  
- **Background listening** with `speech_recognition`  
- Supports CUDA (GPU) and CPU fallback  

---

## 📦 Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/yourusername/spk-verification.git
cd spk-verification

python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)

pip install -r requirements.txt
```

### Requirements
- Python 3.9+  
- [ffmpeg](https://ffmpeg.org/download.html) installed and available in PATH  
- A working microphone  

### Requirements.txt
```
torch
numpy
scipy
speechrecognition
pyannote.audio
faster-whisper
python-dotenv
```

---

## ⚙️ Setup

### 1. HuggingFace Token & Model
Speaker verification requires the [pyannote/embedding](https://huggingface.co/pyannote/embedding) model.  

This model is **gated**, so you need to:

1. Create a free [HuggingFace account](https://huggingface.co/join).  
2. Go to [pyannote/embedding](https://huggingface.co/pyannote/embedding) and **accept the terms**.  
3. Create an **access token** in your HuggingFace settings.  
4. Save it in a `.env` file at the project root:

```
HF_API_KEY=hf_yourapikeyhere
```

The model will be automatically downloaded into `models/pyannote/` on first run.

---

### 2. Create Your Reference Speaker File
You must create your own `.wav` file that contains a short recording of the **authorized speaker’s voice**.

- Record a few seconds of your voice (e.g. saying your name or a short sentence).  
- Save it as `yourname.wav` in the project root.  
- This file will be used to verify whether the person speaking into the microphone is the same as the reference speaker.  

### Example project root:
```
spk-verification/
│── yourname.wav      # your reference speaker audio file
│── stt.py
│── main.py
│── .env
```

In `stt.py`, update the filename if needed:

```python
self.main_speaker_embedding = self.inference("yourname.wav")
```

---

## ▶️ Usage

Run the main script:

```bash
python main.py
```

### Expected flow:
1. Loads the reference embedding from your `.wav` file.  
2. Listens to the microphone.  
3. When you speak:  
   - If the voice matches → transcribe with Whisper.  
   - If not → prints “Speaker not verified”.  

---

## 📂 Project Structure
```
spk-verification/
│── main.py          # Entry point
│── stt.py           # WhisperSTT class (speaker verification + transcription)
│── yourname.wav     # Reference speaker audio (you create this file)
│── models/pyannote/ # HuggingFace pyannote embedding model (auto-downloaded)
│── .env             # HuggingFace API key
│── requirements.txt
│── README.md
```

---

## 🔧 Configuration

- Change the reference `.wav` file in `stt.py`.  
- Change Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v2`):

```python
WhisperSTT(model_size="base")
```

- Adjust the **speaker verification threshold** (`0.7`) in `speaker_verified()` if needed.  

---

## 🚀 Notes

- On **GPU**: `faster-whisper` will use `float16` by default.  
- On **CPU**: falls back to `int8` for speed.  
- First run may take longer as models are downloaded.  
- Ensure your `.wav` reference file is clear and recorded in a quiet environment for best results.  

---

## 📜 License
MIT — use this freely in your own projects.

