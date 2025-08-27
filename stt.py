import os
import time
import torch
import numpy as np
from scipy.spatial.distance import cdist
import speech_recognition as sr
from pyannote.audio import Model, Inference
from faster_whisper import WhisperModel


class WhisperSTT:
    def __init__(self, model_size="base"):
        # torch.device for pyannote
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Speaker embedding model (pyannote)
        embeddingmodel = Model.from_pretrained(
            checkpoint="pyannote/embedding",
            cache_dir="models/pyannote",
            use_auth_token=os.getenv("HF_API_KEY")
        )
        self.inference = Inference(embeddingmodel, window="whole", device=self.device)
        self.main_speaker_embedding = self.inference("yourname.wav")

        # Faster-Whisper model
        whisper_device = self.device.type  # "cuda" or "cpu"
        compute_type = "float16" if whisper_device == "cuda" else "int8"

        self.whisper = WhisperModel(
            model_size,
            device=whisper_device,
            compute_type=compute_type
        )

    def callback(self, recognizer, audio):
        try:
            transcribed_file = "transcribed-audio.wav"
            with open(transcribed_file, "wb") as f:
                f.write(audio.get_wav_data())

            start = time.time()
            output = self.process_audio(transcribed_file)
            end = time.time()
            print(f" Listen Offline Transcription in {end - start:.2f} seconds")

            if output.strip():
                print("Hey [yourname], how are you doing? You said... " + output)

        except sr.UnknownValueError:
            print("Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results from Speech Recognition service: {e}")
        except Exception as e:
            print(f"Error during transcription: {e}")

    def listen(self):
        print("Listening (mode: Faster-Whisper)...")
        try:
            r = sr.Recognizer()
            source = sr.Microphone(sample_rate=16000)

            print("Say something!")
            stop_listening = r.listen_in_background(source, self.callback)

            while True:
                time.sleep(0.05)

        except Exception as e:
            print(f"Error while listening: {e}")
            stop_listening(wait_for_stop=False)
            return False

    def speaker_verified(self, speaker_wav) -> bool:
        speaker_embedding = self.inference(speaker_wav)

        distance = cdist(
            np.reshape(self.main_speaker_embedding, (1, -1)),
            np.reshape(speaker_embedding, (1, -1)),
            metric="cosine"
        )[0, 0]

        print("Speaker Cosine Distance:", distance)
        return distance < 0.7  # threshold

    def process_audio(self, wav_file, model_lang="en") -> str:
        if not self.speaker_verified(wav_file):
            print("Speaker not verified")
            return ""

        segments, info = self.whisper.transcribe(wav_file, language=model_lang)

        # Concatenate text from all segments
        transcription = " ".join(segment.text for segment in segments).strip()

        # Clean noisy tokens if any
        transcription = (
            transcription.replace("[BLANK_AUDIO]", "")
            .replace("[INAUDIBLE]", "")
            .replace("[ Silence ]", "")
            .replace("[Silence]", "")
            .strip()
        )

        return transcription
