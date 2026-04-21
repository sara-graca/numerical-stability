import os
import json
import yaml
import numpy as np
import pandas as pd
import torch
import torchaudio
import soundfile as sf
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from tqdm import tqdm

INPUT_CSV         = os.path.join("data", "parsed", "words.csv")
USABLE_WORDS_PATH = os.path.join("data", "parsed", "usable_words.json")
OUTPUT_FILE       = os.path.join("data", "features", "word_embeddings.parquet")

with open("params.yaml") as f:
    params = yaml.safe_load(f)

MODEL_NAME = params["extract"]["model"]
TARGET_SR  = params["extract"]["target_sr"]
MIN_FRAMES = params["extract"]["min_frames"]

# load usable words and filter dataset

with open(USABLE_WORDS_PATH, "r", encoding="utf-8") as f:
    usable_words = set(json.load(f))

df = pd.read_csv(INPUT_CSV)
df = df[df["word"].isin(usable_words)].reset_index(drop=True)

if df.empty:
    raise RuntimeError("No rows after filtering to usable words. Re-run parse_corpus.py first.")

print(f"Rows after filtering to usable words: {len(df)}")

# load model

device    = "cuda" if torch.cuda.is_available() else "cpu"
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model     = Wav2Vec2Model.from_pretrained(MODEL_NAME)
model     = model.to(device)
model.eval()

# audio cache

audio_cache: dict[str, torch.Tensor] = {}

def load_audio(path: str) -> torch.Tensor:
    if path not in audio_cache:
        data, sr = sf.read(path)
        wav = torch.from_numpy(data).float()

        # soundfile returns (samples, channels) for stereo — average over channels axis (dim=1)
        if wav.ndim > 1:
            wav = wav.mean(dim=1)

        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)

        audio_cache[path] = wav
    return audio_cache[path]

# extract embeddings

embeddings = []
skipped    = 0

for _, row in tqdm(df.iterrows(), total=len(df)):
    wav     = load_audio(row["wav_path"])
    start   = int(row["onset"]  * TARGET_SR)
    end     = int(row["offset"] * TARGET_SR)
    segment = wav[start:end]

    if len(segment) < MIN_FRAMES:
        skipped += 1
        continue

    inputs       = processor(segment, sampling_rate=TARGET_SR, return_tensors="pt")
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        hidden_states = model(input_values).last_hidden_state.squeeze(0)

    word_embedding = hidden_states.mean(dim=0).cpu().numpy()

    embeddings.append({
        "word":          row["word"],
        "speaker_id":    row["speaker_id"],
        "sent_id":       row["sent_id"],
        "sentence_text": row["sentence_text"],
        "embedding":     word_embedding,
    })

if not embeddings:
    raise RuntimeError("No embeddings extracted. Check MIN_FRAMES and audio paths.")

print(f"Extracted {len(embeddings)} embeddings, skipped {skipped} short segments.")

# save

out_df     = pd.DataFrame(embeddings)
emb_matrix = np.stack(out_df["embedding"].values).astype(np.float64) # needed to observe difference between computed distances
emb_cols   = [f"dim_{i}" for i in range(emb_matrix.shape[1])]
emb_df     = pd.DataFrame(emb_matrix, columns=emb_cols)
final_df   = pd.concat([out_df.drop(columns=["embedding"]), emb_df], axis=1)

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
final_df.to_parquet(OUTPUT_FILE, index=False)

print(f"Saved embeddings → {OUTPUT_FILE}")
print(f"Shape: {final_df.shape}")