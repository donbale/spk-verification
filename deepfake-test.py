import os
import numpy as np
from scipy.spatial.distance import cdist
from pyannote.audio import Model, Inference
import matplotlib.pyplot as plt
import csv
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# ------------------------
# Config
# ------------------------
REFERENCE_FILE = "yourname.wav"     # your real reference voice
TEST_DIR = "test_samples"        # folder with deepfake + real wavs
THRESHOLD = 0.7                  # lower distance = more similar

# ------------------------
# Helper: normalize embeddings
# ------------------------
def get_embedding(inference, wav_file):
    emb = inference(wav_file)
    emb = np.array(emb)
    if emb.ndim > 1:  # (frames, features)
        emb = np.mean(emb, axis=0)  # average across frames
    return emb

# ------------------------
# Load model + embeddings
# ------------------------
HF_TOKEN = os.getenv("HF_API_KEY")
if HF_TOKEN is None:
    raise ValueError("Please set HF_API_KEY in your .env file")

model = Model.from_pretrained("pyannote/embedding", use_auth_token=HF_TOKEN)
inference = Inference(model)

# Reference embedding
print(f"Loading reference voice: {REFERENCE_FILE}")
ref_embedding = get_embedding(inference, REFERENCE_FILE)

# ------------------------
# Test all files in TEST_DIR
# ------------------------
print("\n=== Running Speaker Verification Tests ===\n")
results = []

for f in os.listdir(TEST_DIR):
    if not f.endswith(".wav"):
        continue

    path = os.path.join(TEST_DIR, f)
    test_embedding = get_embedding(inference, path)

    distance = cdist(
        np.reshape(ref_embedding, (1, -1)),
        np.reshape(test_embedding, (1, -1)),
        metric="cosine"
    )[0, 0]

    accepted = distance < THRESHOLD
    console_result = "ACCEPTED ✅" if accepted else "REJECTED ❌"
    csv_result = "ACCEPTED" if accepted else "REJECTED"

    results.append((f, distance, csv_result))
    print(f"{f:25s} | distance={distance:.3f} | {console_result}")

# ------------------------
# Save results to CSV (safe format)
# ------------------------
with open("deepfake_results.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["File", "Cosine Distance", "Result"])
    writer.writerows(results)

print("\nResults saved to deepfake_results.csv")

# ------------------------
# Plot results
# ------------------------
files = [r[0] for r in results]
distances = [r[1] for r in results]
colors = ["green" if d < THRESHOLD else "red" for d in distances]

plt.figure(figsize=(10, 6))
bars = plt.bar(files, distances, color=colors)
plt.axhline(THRESHOLD, color="blue", linestyle="--", label=f"Threshold ({THRESHOLD})")

# Add labels above bars
for bar, d in zip(bars, distances):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{d:.2f}", ha="center", va="bottom")

plt.title("Speaker Verification: Real vs Deepfake Voices")
plt.xlabel("Audio Sample")
plt.ylabel("Cosine Distance (lower = more similar)")
plt.xticks(rotation=30, ha="right")
plt.legend()
plt.tight_layout()
plt.savefig("deepfake_results.png", dpi=300)
plt.show()